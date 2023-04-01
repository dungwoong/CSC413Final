# Shufflenet https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
# FastGAN https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py

from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def conv2d(*args, **kwargs):
    # The FastGAN paper used spectral normalization, but I don't think we need that for a convnet.
    # return spectral_norm(nn.Conv2d(*args, **kwargs))
    return nn.Conv2d(*args, **kwargs)


class Swish(nn.Module):
    """
    Also known as SiLU --> sigmoid linear unit

    Applies sigmoid to itself then multiplies
    """

    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SLEBlock(nn.Module):
    """
    SLE block
    """

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                  # they used SiLU instead of LeakyReLU
                                  nn.Conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                  nn.Conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
                                  nn.Sigmoid())
        self.ch_in, self.ch_out = ch_in, ch_out
        self.flops = None

    def calculate_flops(self, big_size, small_size):
        # TODO not counting adaptive pooling. Flops count both mult. and addition
        self.flops = 0
        self.flops += 16 * self.ch_in * self.ch_out * 2  # first conv2d
        self.flops += self.ch_out * 2  # swish, 1 per sigmoid(not accurate) + 1 for mult
        self.flops += self.ch_out * self.ch_out * 2  # second conv2d
        self.flops += self.ch_out  # sigmoid (not accurate)
        self.flops += big_size * big_size  # element wise mult for excitation of second map

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


# SHUFFLENET #############################################################################################

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """
    Channel shuffle operation
    :param x: The tensor to shuffle (B C H W)
    :param groups: The number of groups to shuffle between
    :return: channel shuffled tensor, of same size as the original.
    """
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class SEBlock(nn.Module):

    def __init__(self, oup, reduction):
        """

        :param oup: features
        :param reduction: reduction factor
        """
        super().__init__()
        mid_channels = oup // reduction
        self.se_block = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1),
                                      nn.Conv2d(oup, mid_channels, kernel_size=1, bias=True),
                                      # conv1x1 works with the matrix shape better
                                      nn.ReLU(),
                                      nn.Conv2d(mid_channels, oup, kernel_size=1, bias=True),
                                      nn.Sigmoid()
                                      )
        self.oup = oup
        self.mid_channels = mid_channels

        self.flops = None

    def calculate_flops(self, input_size):
        self.flops = 0
        self.flops += self.oup * self.mid_channels * 2  # first conv
        self.flops += self.mid_channels  # relu
        self.flops += self.oup * self.mid_channels * 2  # second conv
        self.flops += self.oup  # sigmoid, not accurate
        self.flops += input_size * input_size * self.oup  # element-wise multiplication for excitation

    def forward(self, x):
        w = self.se_block(x)
        return w * x


class InvertedResidual(nn.Module):
    """
    A residual block used in ShufflenetV2.
    """

    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        if (self.stride == 1) and (inp != branch_features << 1):
            raise ValueError(
                f"Invalid combination of stride {stride}, inp {inp} and oup {oup} values. If stride == 1 then inp should be equal to oup // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
            i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(
            self,
            stages_repeats: List[int],
            stages_out_channels: List[int],
            num_classes: int = 1000,
            inverted_residual: Callable[..., nn.Module] = InvertedResidual,
            se: bool = False,
            stages_reductions: List = [0, 0, 0],  # reduction for 2, 3, 4
            label='ShuffleNetV2'
    ) -> None:
        # 01234 is conv1, stage2, stage3, stage4, conv5
        super().__init__()
        self.label = label

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels
        if len(stages_reductions) != 3:
            raise ValueError("expected stages_reductions as a list of 3 positive ints")
        self._stages_reductions = stages_reductions if stages_reductions else None

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        # repeats the residual blocks here for each stage
        idx = 0
        stage_names = [f"stage{i}" for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]

            # [ADDITION] added SE block
            if se:
                seq.append(SEBlock(output_channels, self._stages_reductions[idx]))
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))

                # [ADDITION] added SE block
                if se:
                    seq.append(SEBlock(output_channels, self._stages_reductions[idx]))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
            idx += 1

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)

        self.se_1 = SLEBlock(self._stage_out_channels[0], self._stage_out_channels[1])  # maxpool and stage2
        self.se_2 = SLEBlock(self._stage_out_channels[1], self._stage_out_channels[2])  # stage2 to stage3
        self.se_3 = SLEBlock(self._stage_out_channels[2], self._stage_out_channels[3])  # stage3 to stage4

    def get_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        for k in self.sles:
            t = sum(p.numel() for p in self.sles[k].parameters())
            tr = sum(p.numel() for p in self.sles[k].parameters() if p.requires_grad)
            total += t
            trainable += tr
            # print(f"SLE got {tr} trainable params")
        return trainable, total

    def _forward_SLE(self, x: Tensor) -> Tensor:
        p1 = self.conv1(x)
        # p1 = self.maxpool(x)
        stage2 = self.stage2(p1)
        stage2 = self.se_1(p1, stage2)
        stage3 = self.stage3(stage2)
        stage3 = self.se_2(stage2, stage3)
        stage4 = self.stage4(stage3)
        stage4 = self.se_3(stage3, stage4)
        final_conv = self.conv5(stage4)
        final_conv = final_conv.mean([2, 3])  # global pool along H W
        return self.fc(final_conv)
    #
    # def _forward_impl(self, x: Tensor) -> Tensor:
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.maxpool(x)
    #     x = self.stage2(x)
    #     x = self.stage3(x)
    #     x = self.stage4(x)
    #     x = self.conv5(x)
    #     x = x.mean([2, 3])  # globalpool
    #     x = self.fc(x)
    #     return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # return self._forward_impl(x)
        return self._forward_SLE(x, **kwargs)


STAGES_REPEATS = [4, 8, 4]
STAGES_OUT_CHANNELS_1 = [24, 116, 232, 464, 1024]
STAGES_OUT_CHANNELS_1_5 = [24, 176, 352, 704, 1024]


# def base_model(device):
#     return ShuffleNetV2(stages_repeats=STAGES_REPEATS,
#                         stages_out_channels=STAGES_OUT_CHANNELS_1,
#                         label="ShuffleNetV2").to(device)
#
#
# def se_model(device):
#     return ShuffleNetV2(stages_repeats=STAGES_REPEATS,
#                         stages_out_channels=STAGES_OUT_CHANNELS_1,
#                         se=True,
#                         stages_reductions=[8, 16, 16],
#                         label="ShuffleNetV2+SE").to(device)


def sle_model_countable(device):
    s = ShuffleNetV2(stages_repeats=STAGES_REPEATS,
                     stages_out_channels=STAGES_OUT_CHANNELS_1,
                     label="ShuffleNetV2+SLE").to(device)
    return s

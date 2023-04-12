# Shufflenet https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
# FastGAN https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
# This code was not used for the final project. We opted for a different implementation

from functools import partial
import math
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
            num_classes: int = 10,  # cifar10
            inverted_residual: Callable[..., nn.Module] = InvertedResidual,
            se: bool = False,
            stages_reductions: List = [0, 0, 0],  # reduction for 2, 3, 4
            sle_config: List[Tuple] = [],  # (from, to) list, starting at 0
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

        # self.sles = dict()
        # for from_layer, to_layer in sle_config:
        #     assert 0 <= from_layer < to_layer < 5, "SLEs must connect to future stages, stages are 0-4"
        #     self.sles[from_layer, to_layer] = SLEBlock(self._stage_out_channels[from_layer],
        #                                                self._stage_out_channels[to_layer])

    # def move_sles(self, device):
    #     for k in self.sles:
    #         self.sles[k] = self.sles[k].to(device)
    #
    # # [ADDITION] New forward method with SLEs
    # def _new_forward(self, x):
    #     outputs = [None, None, None, None, None]  # stages 1 to 5, I guess 0 to 4
    #     outputs[0] = self.conv1(x)
    #     outputs[1] = self.stage2(outputs[0])
    #     self.add_sle(outputs, 1)
    #     outputs[2] = self.stage3(outputs[1])
    #     self.add_sle(outputs, 2)
    #     outputs[3] = self.stage4(outputs[2])
    #     self.add_sle(outputs, 3)
    #     outputs[4] = self.conv5(outputs[3])
    #     self.add_sle(outputs, 4)
    #     final_conv = outputs[4].mean([2, 3])  # global avg pool
    #     return self.fc(final_conv)
    #
    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _forward_impl(self, x):
        s1 = self.conv1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.conv5(s4)
        final_conv = s5.mean([2, 3])  # global avg pool
        return self.fc(final_conv)

    #
    # # [ADDITION] Adding SLE excitation for each stage. Note nothing will happen if self.sles is empty dictionary.
    # def add_sle(self, outputs, layer):
    #     """
    #     Applies SLE on the current layer output.
    #
    #     Supports 1 SLE going to each stage, from a previous stage btw
    #     :param outputs: list of 5 things, representing output of each stage
    #     :param layer: int, current layer we are considering
    #     :return:
    #     """
    #     for from_layer, to_layer in self.sles:
    #         if to_layer == layer:
    #             outputs[layer] = self.sles[(from_layer, to_layer)](outputs[from_layer], outputs[to_layer])
    #             return

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        # return self._forward_impl(x)
        return self._forward_impl(x, **kwargs)


class ShuffleNetSLE(ShuffleNetV2):
    def __init__(
            self,
            *args, **kwargs
    ) -> None:
        # 01234 is conv1, stage2, stage3, stage4, conv5
        super().__init__(*args, **kwargs)

        self.se_1 = SLEBlock(self._stage_out_channels[0], self._stage_out_channels[1])  # maxpool and stage2
        self.se_2 = SLEBlock(self._stage_out_channels[1], self._stage_out_channels[2])  # stage2 to stage3
        self.se_3 = SLEBlock(self._stage_out_channels[2], self._stage_out_channels[3])  # stage3 to stage4

    def _forward_SLE(self, x: Tensor) -> Tensor:
        p1 = self.relu(self.conv1(x))
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

    def forward(self, x):
        return self._forward_SLE(x)


STAGES_REPEATS = [4, 8, 4]
STAGES_OUT_CHANNELS_0_5 = [24, 48, 96, 192, 1024]
STAGES_OUT_CHANNELS_1 = [24, 116, 232, 464, 1024]
STAGES_OUT_CHANNELS_1_5 = [24, 176, 352, 704, 1024]


def base_model(**kwargs):
    s = ShuffleNetV2(stages_repeats=STAGES_REPEATS,
                     stages_out_channels=STAGES_OUT_CHANNELS_0_5,
                     label="ShuffleNetV2",
                     **kwargs)
    s.weights_init()
    return s


def se_model(**kwargs):
    s = ShuffleNetV2(stages_repeats=STAGES_REPEATS,
                     stages_out_channels=STAGES_OUT_CHANNELS_1,
                     se=True,
                     stages_reductions=[8, 16, 16],
                     label="ShuffleNetV2_SE",
                     **kwargs)
    s.weights_init()
    return s


# def sle_model(device):
#     s = ShuffleNetV2(stages_repeats=STAGES_REPEATS,
#                      stages_out_channels=STAGES_OUT_CHANNELS_1,
#                      sle_config=[(1, 2), (2, 3), (3, 4)],
#                      label="ShuffleNetV2+SLE").to(device)
#     s.move_sles(device)
#     return s

def sle_model(**kwargs):
    s = ShuffleNetSLE(stages_repeats=STAGES_REPEATS,
                      stages_out_channels=STAGES_OUT_CHANNELS_1,
                      label="ShuffleNetV2_SLE",
                      **kwargs)
    s.weights_init()
    return s

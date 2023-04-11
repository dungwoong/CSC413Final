import argparse
import time

import pandas as pd
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import TensorDataset

from shufflenet_alt import ShuffleNetV2, ShuffleNetSE, ShuffleNetSLE, init_params
from util import top1_error, top3_error, plot_training_curve

mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, std)
])
transform_train = torchvision.transforms.Compose([
      torchvision.transforms.RandomCrop(32, padding = 4),
      torchvision.transforms.RandomHorizontalFlip(),
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize(mean, std)])

train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
train_size = len(train_set)
test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
test_size = len(test_set)

print(f"Training data has {train_size} observations, test has {test_size}.")


# code adapted from https://colab.research.google.com/github/uoft-csc413/2023/blob/master/assets/tutorials/tut04_cnn.ipynb#scrollTo=Ztj0yQO8-TtS

def get_dataloaders(batch_size, test_bsize=64):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_bsize, shuffle=False, num_workers=1)

    data_loaders = {"train": train_loader, "test": test_loader}
    dataset_sizes = {"train": train_size, "test": test_size}
    return data_loaders, dataset_sizes


def run_epoch(model, loss_fn, optimizer, device, data_loaders, dataset_sizes):
    epoch_loss = {"train": 0.0, "test": 0.0}
    epoch_acc_1 = {"train": 0.0, "test": 0.0}
    epoch_acc_5 = {"train": 0.0, "test": 0.0}  # 5 is actually 3 btw

    # running loss for train/test phase
    running_loss = {"train": 0.0, "test": 0.0}
    running_corrects_1 = {"train": 0, "test": 0}
    running_corrects_5 = {"train": 0, "test": 0}

    # time and batch data
    start_time = time.time()
    batches = {"train": 0, "test": 0}

    for phase in ["train", "test"]:
        print(f"Running phase {phase}")
        # set train/eval mode
        if phase == "train":
            model.train(True)
        else:
            model.train(False)

        # go thru batches
        for data in data_loaders[phase]:
            batches[phase] += 1
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # clear all gradients

            outputs = model(inputs)  # batch_size x num_classes
            sm_outputs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs.data, 1)  # values, indices
            loss = loss_fn(outputs, labels)

            if phase == "train":
                loss.backward()  # compute gradients
                optimizer.step()  # update weights/biases

            running_loss[phase] += loss.data.item() * inputs.size(0)
            c, t = top1_error(sm_outputs, labels.data)
            running_corrects_1[phase] += c
            c2, t2 = top3_error(sm_outputs, labels.data)
            running_corrects_5[phase] += c2

            epoch_loss[phase] = running_loss[phase] / dataset_sizes[phase]
            epoch_acc_1[phase] = running_corrects_1[phase] / dataset_sizes[phase]
            epoch_acc_5[phase] = running_corrects_5[phase] / dataset_sizes[phase]

    return {"loss": epoch_loss,
            "top1_acc": epoch_acc_1,
            "top3_acc": epoch_acc_5,
            "running_corrects_1": running_corrects_1,
            "running_corrects_3": running_corrects_5,
            "dataset_sizes": dataset_sizes,
            "time": time.time() - start_time,
            "batches": batches}


def flatten_dict(dic, sep='_'):
    ret = dict()
    for key in dic:
        if isinstance(dic[key], dict):
            flat = flatten_dict(dic[key], sep='_')
            for key2 in flat:
                ret[key + sep + key2] = flat[key2]
        else:
            ret[key] = dic[key]
    return ret


def to_df(flattened_dict):
    d = {key: [flattened_dict[key]] for key in flattened_dict}
    return pd.DataFrame(d)


def train(model, device, batch_size, lr, beta0, beta1, weight_decay, checkpoint=None, epochs=100,
          lr_decay_rate=10, lr_decay_epochs=[], decay_patience=10, csv_path="", models_path="tmp/", plot=True,
          print_results_every_epoch=False):
    # lr_decay_rate will apply every lr_decay_epochs epochs
    results = None

    # save model info
    label = model.label if hasattr(model, "label") else "ShuffleNetV2"
    res_csv = f"{csv_path}{label}_results.csv"
    mod_csv = f"{csv_path}{label}_params.csv"
    model_info = {"keys": ["label", "batch_size", "lr", "beta0", "beta1", "weight_decay", "lr_decay", "lr_decay_freq",
                           "lr_decay_patience"],
                  "values": [label, batch_size, lr, beta0, beta1, weight_decay, lr_decay_rate, lr_decay_epochs,
                             decay_patience]}
    print(f"Saving model info to {mod_csv}")
    model_info = pd.DataFrame(model_info)
    model_info.to_csv(mod_csv, index=False)

    data_loaders, dataset_sizes = get_dataloaders(batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), betas=(beta0, beta1), lr=lr, weight_decay=weight_decay)

    # [NEW] Define a learning rate scheduler to decrease the learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1/lr_decay_rate, patience=decay_patience)

    # train for many epochs
    for i in range(epochs):
        print(f"Epoch {i + 1} / {epochs}")
        print("-" * 30)
        epoch_res = run_epoch(model, loss_fn, optimizer, device, data_loaders, dataset_sizes)
        epoch_res["epoch"] = i
        if print_results_every_epoch:
            print(epoch_res)
        epoch_flat = flatten_dict(epoch_res)
        epoch_res = to_df(epoch_flat)
        # print(epoch_res.transpose()) # print the flat version?
        results = pd.concat([results, epoch_res], axis=0) if results is not None else epoch_res

        # save information
        print(f"Saving to {res_csv}")
        results.to_csv(res_csv, index=True)
        model_info.to_csv(mod_csv, index=False)

        # save model info
        epoch_formatted = '{:04d}'.format(i)
        torch.save({"mod": model.state_dict(),
                    "opt": optimizer.state_dict()}, f"{models_path}{epoch_formatted}.pth")

        if plot:
            plot_training_curve(results, save=True, save_path=f"{csv_path}curve")

        # scheduler.step(epoch_flat["loss_train"])

        if i > 0 and i in lr_decay_epochs:
            print("Decreasing LR...")
            lr /= lr_decay_rate
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta0, beta1), weight_decay=weight_decay)
    return results, model_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net",
        type=str,
        required=True,
        help="base/se/sle"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=True
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    torch.manual_seed(1234) # recently added, older results are not reproducible
    # model = se_model().to(device) # updated model file to not include device
    if args.net == "base":
        print("Using Base model")
        model = ShuffleNetV2(net_size=1)
    elif args.net == "se":
        print("Using SE")
        model = ShuffleNetSE(net_size=1)
    else:
        print("Using SLE")
        model = ShuffleNetSLE(net_size=1)

    init_params(model)
    model = model.to(device)
    # 5e-3 is pretty effective for SE model
    models_path = "ShuffleNetV2/models/01/"
    res_path = "ShuffleNetV2/results/01/"
    train(model, device, args.batch_size, args.lr, beta0=0.9, beta1=0.999, weight_decay=1e-4,
          epochs=args.epochs, lr_decay_rate=10, lr_decay_epochs=[], csv_path=args.csv, models_path=args.models)
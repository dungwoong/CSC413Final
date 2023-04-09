import argparse
import os
import pandas as pd


def get_directories(root):
    # initialize an empty list to store the directories
    directories = []
    # loop through all items in the root directory
    for item in os.listdir(root):
        # construct the full path to the item
        path = os.path.join(root, item)
        # check if the item is a directory
        if os.path.isdir(path):
            # if it is a directory, check if it contains any CSV files
            if any(item.endswith('.csv') for item in os.listdir(path)):
                # if it contains CSV files, add it to the list of directories
                directories.append(path)
            else:
                # if it does not contain CSV files, call the function recursively
                directories.extend(get_directories(path))
    return directories


def loop_through_files(directory):
    # loop through all items in the directory
    paths = []
    for item in os.listdir(directory):
        # construct the full path to the item
        path = os.path.join(directory, item)
        if "ipynb" in path:
            continue
        # check if the item is a file
        if os.path.isfile(path) and "csv" in path:
            # if it is a file, do something with it
            print("Found file:", path)
            paths.append(path)

    assert len(paths) == 2, f"{directory} expected 2 files"
    if 'params' in paths[0]:
        return paths[0], paths[1]
    return paths[1], paths[0]


def get_params(path):
    df = pd.read_csv(path)
    lr_value = df.loc[df['keys'] == 'lr', 'values'].iloc[0]
    batch_size_value = df.loc[df['keys'] == 'batch_size', 'values'].iloc[0]
    return lr_value, batch_size_value


def get_val_stats(path):
    # only consider first 100 epochs
    df = pd.read_csv(path, nrows=100)

    # we might want the indices later too so idk i'll just keep this here
    # get the index of the row with the minimum 'loss_test'
    min_loss_index = df['loss_test'].idxmin()

    # get the index of the row with the maximum 'top1_acc_test'
    max_acc_index = df['top1_acc_test'].idxmax()

    # get the values of the minimum 'loss_test' and maximum 'top1_acc_test'
    min_loss_value = df.loc[min_loss_index, 'loss_test']
    max_acc_value = df.loc[max_acc_index, 'top1_acc_test']

    # return the values as a tuple
    result = (min_loss_value, max_acc_value)
    return result


def run(root, outputfile="aggregated.csv", model_label=""):
    df = None
    d = get_directories(root)
    print("D", d)
    for flder in d:
        param_file, results_file = loop_through_files(flder)
        lr, bs = get_params(param_file)
        min_loss, max_acc = get_val_stats(results_file)
        stats = {"model": [model_label],
                 "lr": [lr],
                 "batch_size": [bs],
                 "min_val_loss": [min_loss],
                 "max_val_top1_acc": [max_acc]}
        tmp = pd.DataFrame(stats)
        if df is None:
            df = tmp
        else:
            df = pd.concat([df, tmp])
    df.to_csv(outputfile, index=False)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--root",
    #     type=str,
    #     required=True,
    #     help="Root dir for the model results"
    # )
    # parser.add_argument(
    #     "--o",
    #     type=str,
    #     required=False,
    #     default=None,
    #     help="output file"
    # )
    # args = parser.parse_args()
    run("results/ShuffleNetSE", model_label="SLE")

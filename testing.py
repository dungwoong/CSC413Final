from shufflenet_alt import ShuffleNetV2, ShuffleNetSE, ShuffleNetSLE
import torch
from ptflops import get_model_complexity_info
import pandas as pd


def get_device():
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    return device


def test_model_works(mod, device, batch_size=1):
    print(f"Testing {mod.__class__}")
    if device == 'cuda':
        torch.cuda.empty_cache()
    # basic test, makes sure model's forward() method actually works
    test_example = torch.randn((batch_size, 3, 32, 32)).to(device)
    mod(test_example)


def get_params(mods, outputpath):
    ret = {"Model": [],
           "Parameters": [],
           "MMac": []}
    for mod in mods:
        macs, params = get_model_complexity_info(mod, (3, 32, 32), as_strings=False, verbose=False)
        ret['Model'].append(mod.__class__)
        ret['Parameters'].append(params)
        ret['MMac'].append(macs)
    d = pd.DataFrame(ret)
    d.to_csv(outputpath, index=False)


if __name__ == '__main__':
    d = get_device()
    get_params([ShuffleNetV2(net_size=1),
                ShuffleNetSE(net_size=1),
                ShuffleNetSLE(net_size=1)], outputpath="model_stats.csv")

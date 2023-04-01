import shufflenetv2
import torch


def get_device():
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    return device


def test_model_works(mod, device, batch_size=1):
    print(f"Testing {mod.label}")
    if device == 'cuda':
        torch.cuda.empty_cache()
    # basic test, makes sure model's forward() method actually works
    test_example = torch.randn((batch_size, 3, 32, 32)).to(device)
    mod(test_example)


def report_params(mod):
    trainable, total = mod.get_params()
    print(f"{mod.label}: {trainable} trainable parameters, {total} total parameters")


if __name__ == '__main__':
    d = get_device()
    report_params(shufflenetv2.base_model(d))
    report_params(shufflenetv2.se_model(d))
    report_params(shufflenetv2.sle_model(d))
    # test_model_works(shufflenetv2.base_model(d), d)
    # test_model_works(shufflenetv2.se_model(d), d)
    # test_model_works(shufflenetv2.sle_model(d), d)

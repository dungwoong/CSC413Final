import torch


def top5_error(predictions, targets):
    with torch.no_grad():
        _, top5_pred = torch.topk(predictions, k=5, dim=1)
        top5_correct = top5_pred.eq(targets.expand(top5_pred.size()))
        correct = top5_correct.float().sum()
        total = targets.size(0)
    return correct, total


def top1_error(predictions, targets):
    with torch.no_grad():
        _, top1_pred = torch.max(predictions, dim=1)
        top1_correct = top1_pred.unsqueeze(1).eq(targets)
        correct = top1_correct.float().sum()
        total = targets.size(0)
    return correct, total

# top5 should be 100%, top1 should be ~20%
# preds = torch.randn((1024, 5))
# targets = torch.ones((1024, 1))
# c, t = top5_error(preds, targets)
# print(f"Top 5: {c/t}, ({c} / {t})")
# c, t = top1_error(preds, targets)
# print(f"Top 1: {c/t}, ({c} / {t})")

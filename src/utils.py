import torch

def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if len(y) == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    return correct / len(y)
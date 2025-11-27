import torch


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    if len(y) == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == y).sum().item()
    return correct / len(y)


def f1_macro(logits, y_true, num_classes=None):
    """
    Calcula F1 macro para clasificación multiclase.
    logits: tensor [N, C] con scores o logits
    y_true: tensor [N] con labels verdaderas
    """
    preds = logits.argmax(dim=-1)

    if num_classes is None:
        num_classes = logits.size(1)

    f1_per_class = []
    for c in range(num_classes):
        tp = ((preds == c) & (y_true == c)).sum().item()
        fp = ((preds == c) & (y_true != c)).sum().item()
        fn = ((preds != c) & (y_true == c)).sum().item()

        # Si esta clase no aparece en este split, la omitimos del promedio
        if tp == 0 and fp == 0 and fn == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        f1_per_class.append(f1)

    if not f1_per_class:
        return 0.0

    return float(sum(f1_per_class) / len(f1_per_class))

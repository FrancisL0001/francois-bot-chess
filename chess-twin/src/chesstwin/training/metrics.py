import torch
import torch.nn.functional as F

def masked_cross_entropy(logits, targets, masks):
    logits = logits.masked_fill(~masks, float("-inf"))
    return F.cross_entropy(logits, targets)

@torch.no_grad()
def top1_accuracy(logits, targets, masks):
    logits = logits.masked_fill(~masks, float("-inf"))
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

@torch.no_grad()
def topk_accuracy(logits, targets, masks, k=1):
    logits = logits.masked_fill(~masks, -1e9)          # same legal-mask, finite fill
    topk = logits.topk(k, dim=1).indices               # (B, k) indices of k highest
    hits = (topk == targets.unsqueeze(1)).any(dim=1)    # was the true move among them?
    return hits.float().mean().item()
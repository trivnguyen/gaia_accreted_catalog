
import torch
from tqdm import tqdm


@torch.no_grad()
def infer(model, loader, to_numpy=True, verbose=True):
    model.eval()
    y_pred, y_true = [], []
    if verbose:
        loop = tqdm(loader, desc="Inferencing")
    else:
        loop = loader
    for x, y in loop:
        x = x.to(model.device)
        y = y.to(model.device)
        y_pred.append(model(x).cpu())
        y_true.append(y.cpu())
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    if to_numpy:
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
    return y_pred, y_true

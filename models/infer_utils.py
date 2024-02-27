
import torch
from tqdm import tqdm

@torch.no_grad()
def infer(model, loader, softmax=False, to_numpy=True, verbose=True):
    model.eval()
    y_pred, y_true = [], []
    if verbose:
        loop = tqdm(loader, desc="Inferencing")
    else:
        loop = loader
    for batch in loop:
        batch_dict = model._prepare_training_batch(batch)
        x = batch_dict['x']
        y = batch_dict['y']
        y_pred.append(model(x).cpu())
        y_true.append(y.cpu())
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if softmax:
        y_pred = torch.nn.functional.softmax(y_pred, dim=1)

    if to_numpy:
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
    return y_pred, y_true

def eff(y_true, y_score, thresholds):
    """ Get the efficiency and purity for a given threshold """
    epsilon_0 = []  # efficiency for class 0
    epsilon_1 = []  # efficiency for class 1
    purity = []  # purity for class 1
    for threshold in thresholds:
        y_pred = (y_score > threshold)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        epsilon_0.append(fp / (tn + fp))
        epsilon_1.append(tp / (tp + fn))
        purity.append(tp / (tp + fp))
    return np.array(epsilon_0), np.array(epsilon_1), np.array(purity)

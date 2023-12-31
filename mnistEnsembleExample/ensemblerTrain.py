"""
Optimizer to automatically ensemble models for MNIST dataset
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
import torchvision
import matplotlib.pyplot as plt
from mnistEnsembleExample.cnn1 import CNN1
from mnistEnsembleExample.cnn2 import CNN2
from mnistEnsembleExample.mlp1 import MLP1
from mnistEnsembleExample.mlp2 import MLP2
from mnistEnsembleExample.vit import ViT
from mnistEnsembleExample.xgb import XGBMNIST

torch.backends.cudnn.enabled = False
torch.manual_seed(2)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1000, shuffle=True)

_, (data, targets) = next(enumerate(test_loader))

cnn1 = CNN1().eval()
cnn2 = CNN2().eval()
mlp1 = MLP1().eval()
mlp2 = MLP2().eval()
vit = ViT().eval()
xgb = XGBMNIST()

models = [cnn1, cnn2, mlp1, mlp2, vit, xgb]

n_models = len(models)

l = 0.1  # regularization

with torch.no_grad():
    data_individual = [model(data) for model in models]
    data_all = torch.stack(data_individual)

    def corr(x1, x2):
        return torch.sum((x1-torch.mean(x1))*(x2-torch.mean(x2))) / (x1.shape[0] * x1.shape[1] - 1) / torch.std(x1) / torch.std(x2)

    def corr_correct(X1, X2):
        x1 = torch.tensor([x[cls] for x, cls in zip(X1, targets)]).reshape(-1, 1)
        x2 = torch.tensor([x[cls] for x, cls in zip(X2, targets)]).reshape(-1, 1)
        return corr(x1, x2)

    def nll_corr(X1, X2):
        X1_nll = -torch.log(X1)
        X2_nll = -torch.log(X2)
        return corr_correct(X1_nll, X2_nll)

    correlations = torch.tensor([[nll_corr(x1, x2) for x1 in data_individual] for x2 in data_individual])
    plt.imshow(correlations.numpy())
    plt.colorbar()
    plt.show()

    def fun(weights):
        weights = torch.tensor(weights, dtype=torch.float32)
        weighted_probs = torch.einsum('i,ijk->jk', weights, data_all)
        loss = F.nll_loss(torch.log(weighted_probs), targets).item() + l * weights@weights
        return loss

    def jac(weights):
        weighted_probs = torch.einsum('i,ijk->jk', torch.tensor(weights, dtype=torch.float32), data_all)
        return [-1 / len(targets) * sum(dat[target] / dats[target] for dat, dats, target in zip(dat_ind, weighted_probs, targets)) + 2 * l * weight for dat_ind, weight in zip(data_individual,  weights)]

bounds = [(0, 1)]*n_models


constraints = ({'type': 'eq', 'fun': lambda w: sum(w)-1})


w0 = np.ones(n_models) / n_models

res = minimize(fun, w0, jac=jac, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)

with torch.no_grad():
    _, (data, targets) = next(enumerate(test_loader))
    data_individual = [model(data) for model in models]
    data_all = torch.stack(data_individual)


np.set_printoptions(precision=3)
print(f'Regularization weight: {l}')
print(f'Models used: {[type(model) for model in models]}')
print(f'Weights: {res.x}')
weighted_probs = torch.einsum('i,ijk->jk', torch.tensor(res.x, dtype=torch.float32), data_all)
print(f'Ensemble loss: {F.nll_loss(torch.log(weighted_probs), targets).item():.4f}')
pred = weighted_probs.data.max(1, keepdim=True)[1]
print(f'Ensemble accuracy: {pred.eq(targets.data.view_as(pred)).sum()/len(targets):.4f}')
ind_losses = [f'{F.nll_loss(torch.log(probs), targets).item():.4f}' for probs in data_individual]
print(f'Individual losses: {ind_losses}')
print(f'Individual accuracies: {[probs.data.max(1, keepdim=True)[1].eq(targets.data.view_as(pred)).sum()/len(targets) for probs in data_individual]}')

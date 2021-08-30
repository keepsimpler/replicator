import torch
import torch.nn as nn


class Accumulator:
    """For accumulating sums over `n` variables. Copied from d2l."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


# def add_up_to_non_negative(tensor: torch.Tensor):
#     """
#     Eliminate negative values by adding the minimum value up to zero
#     through by the last dimension.
#     """
#     min_values, _ = tensor.min(dim=-1, keepdim=True)
#     sub_values = torch.minimum(min_values, torch.zeros_like(tensor))
#     return tensor - sub_values


def num_params(net: torch.nn.Module):
    "Sizes of parameters of a Pytorch neural network"
    num_params = 0
    for name, param in net.named_parameters():
        num = torch.prod(torch.tensor(param.size()))
        num_params += num
    return num_params


def net_statistics(net: torch.nn.Module):
    """
    Statistic measures of a Pytorch neural network listed according to named parameters, including:
    - parameters number
    - parameters mean
    - parameters std
    - parameters grad abs mean
    - parameters grad std
    """
    net_statistics = []
    for name, param in net.named_parameters():
        if param.grad is not None:
            net_statistics.append({
                'name': name,
                'num': torch.prod(torch.tensor(param.size())),
                'mean': torch.mean(param.data),
                'std': torch.std(param.data),
                'grad.abs.mean': torch.mean(torch.abs(param.grad)),
                'grad.std': torch.std(param.grad)
            })
    return net_statistics
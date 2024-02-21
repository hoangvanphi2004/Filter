import torch

a = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]);
a[a > 2] = 2;
print(a);
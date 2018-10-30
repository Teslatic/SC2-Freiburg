import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

kwargs = {}

loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transforms=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])batch_size=64, shuffle=True, **kwargs)
)

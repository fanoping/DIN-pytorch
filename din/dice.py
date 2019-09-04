import torch.nn as nn
import torch


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        
        if self.dim == 3:
            self.alpha = torch.zeros((num_features, 1)).cuda()
        elif self.dim == 2:
            self.alpha = torch.zeros((num_features,)).cuda()
        

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        
        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        
        return out
        


if __name__ == "__main__":
    a = Dice(32)
    b = torch.zeros((10, 32))
    #b = torch.transpose(b, 1, 2)
    c = a(b)
    print(c.size())
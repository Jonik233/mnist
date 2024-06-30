from torch import nn

class Net(nn.Module):
    def __init__(self, in_features1, out_features1, in_features2, out_features2, in_features3, out_features3):
        super().__init__()
        self.linear_stack = nn.Sequential(nn.Linear(in_features1, out_features1), 
                                          nn.Tanh(), 
                                          nn.Linear(in_features2, out_features2), 
                                          nn.Tanh(), 
                                          nn.Linear(in_features3, out_features3), 
                                          nn.Tanh()
                                          )

    def forward(self, x):
        output = self.linear_stack(x)
        return output
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_stack = nn.Sequential(nn.Conv1d(28, 28, 5, stride=1, padding=2), 
                                          nn.ReLU(),
                                          nn.Conv1d(28, 28, 5, stride=1, padding=2),
                                          nn.ReLU(),
                                          nn.Conv1d(28, 10, 5, stride=1, padding=2),
                                          nn.ReLU(),
                                          nn.Linear(28, 1)
                                          )

    def forward(self, x):
        output = self.linear_stack(x)
        return output
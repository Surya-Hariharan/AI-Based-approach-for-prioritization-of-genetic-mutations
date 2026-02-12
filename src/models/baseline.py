import torch.nn as nn

class LogisticRegression(nn.Module):
    """
    Baseline Logistic Regression model.
    """
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

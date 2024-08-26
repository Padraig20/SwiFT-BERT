import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_emotions):
        super(SimpleMLP, self).__init__()
        # MLP with two hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_emotions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=4).transpose(1,2) # (b, t, c*h*w*d) = [16, 20, 288*2*2*2]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
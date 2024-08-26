import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, num_emotions):
        super(LinearLayer, self).__init__()
        
        self.fc = nn.Linear(input_dim, num_emotions)
    
    def forward(self, x):
        #(b, c, h, w, d, t) to (b, t, c*h*w*d)
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)
        
        output = self.fc(x)  # Shape: (b, t, num_emotions)
        
        return output

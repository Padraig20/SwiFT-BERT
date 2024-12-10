import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, num_emotions, time_dim):
        super(LinearLayer, self).__init__()
        
        self.time_dim = time_dim
        
        self.temporal_mapping = nn.Linear(time_dim, time_dim)
        self.fc = nn.Linear(input_dim, num_emotions)
    
    def forward(self, x):
        #(b, c, h, w, d, t) to (b, t, c*h*w*d)
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)
        
        if self.time_dim != x.shape[1]:
            x = self.temporal_mapping(x)
        
        output = self.fc(x)  # Shape: (b, t, num_emotions)
                
        return output
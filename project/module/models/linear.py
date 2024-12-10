import torch
import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(self, input_dim, num_emotions, time_dim_orig, time_dim_patched):
        super(LinearLayer, self).__init__()
        
        self.time_dim_patched = time_dim_patched
        self.input_dim = input_dim
        self.time_dim_orig = time_dim_orig
        
        self.temporal_mapping = nn.Linear(time_dim_patched, time_dim_orig, bias=False)
        self.fc = nn.Linear(input_dim, num_emotions)
    
    def forward(self, x):
        #(b, c, h, w, d, t) to (b, t, c*h*w*d)
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)
        
        if self.time_dim_orig != x.shape[1]:
            x = x.permute(0, 2, 1)
            x = self.temporal_mapping(x)
            x = x.permute(0, 2, 1)
        
        output = self.fc(x)  # Shape: (b, t, num_emotions)
                
        return output
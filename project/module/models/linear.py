import torch
import torch.nn as nn

class LinearLayerReg(nn.Module):
    def __init__(self, input_dim, num_emotions):
        super(LinearLayerReg, self).__init__()
        
        self.fc = nn.Linear(input_dim, num_emotions)
    
    def forward(self, x):
        #(b, c, h, w, d, t) to (b, t, c*h*w*d)
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)
        
        output = self.fc(x)  # Shape: (b, t, num_emotions)
        
        return output

class LinearLayerClf(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearLayerClf, self).__init__()
        
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        #(b, c, h, w, d, t) to (b, t, c*h*w*d)
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)
        
        output = self.fc(x)  # Shape: (b, t, num_classes)
        
        #output = torch.sigmoid(output) # Shape: (b, t, num_classes), probabilities [0,1] for each class
        
        return output

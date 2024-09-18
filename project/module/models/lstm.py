import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_emotions):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_emotions)
    
    def forward(self, x):
        # Input shape: (b, c, h, w, d, t)
        
        # Reshape input to (b, t, c*h*w*d)
        b, c, h, w, d, t = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4)  # Move time dimension to the second position: (b, t, c, h, w, d)
        x = x.reshape(b, t, c * h * w * d)  # Flatten spatial dimensions: (b, t, input_dim)
        
        # Pass the reshaped input through the LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (b, t, hidden_dim)
        
        # Apply the fully connected layer at each time step
        output = self.fc(lstm_out)  # Shape: (b, t, num_emotions)
        
        return output

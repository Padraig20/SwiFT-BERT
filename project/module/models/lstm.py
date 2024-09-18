import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_emotions):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_emotions)
    
    def forward(self, x):
        # Shape of input x: (b, c, h, w, d, t)
        
        # Reshape (b, c, h, w, d, t) to (b, t, c*h*w*d)
        b, c, h, w, d, t = x.shape
        x = x.permute(0, 5, 1, 2, 3, 4)  # Move time dimension to the second position: (b, t, c, h, w, d)
        x = x.reshape(b, t, c * h * w * d)  # Flatten spatial dimensions: (b, t, input_dim)
        
        # LSTM expects input of shape (batch_size, sequence_length, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We take the output from the last time step
        last_out = lstm_out[:, -1, :]  # Shape: (b, hidden_dim)
        
        # Pass the output of the last time step to the linear layer
        output = self.fc(last_out)  # Shape: (b, num_emotions)
        
        return output

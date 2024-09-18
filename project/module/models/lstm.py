import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_emotions):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_emotions)
    
    def forward(self, x):
        # reshape (b, c, h, w, d, t) to (b, t, c*h*w*d)
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)
        
        # LSTM expects (batch_size, sequence_length, input_dim)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # we can use the last output from the LSTM (for each time step) or all
        # take the last time step's output
        last_out = lstm_out[:, -1, :]  # Shape: (b, hidden_dim)
        
        output = self.fc(last_out)  # Shape: (b, num_emotions)
        
        return output

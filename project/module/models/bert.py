import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT(nn.Module):
    def __init__(self, num_emotions, hidden_dim, seq_len):
        super(BERT, self).__init__()
        
        custom_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=seq_len,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            vocab_size=30522
        )
        
        self.bert = BertModel(config=custom_config)
        
        self.fc = nn.Linear(custom_config.hidden_size, num_emotions)
    
    def forward(self, x): # (b, c, h, w, d, t) = [16, 288, 2, 2, 2, 20]
        
        x = x.flatten(start_dim=1, end_dim=4).transpose(1,2) # (b, t, c*h*w*d) = [16, 20, 288*2*2*2]
                
        bert_output = self.bert(inputs_embeds=x) #(b,t,hidden_dim)
        
        last_hidden_state = bert_output.last_hidden_state  #(b,t,hidden_dim)
        
        output = self.fc(last_hidden_state)  # Shape: (b,t,e)
        
        return output
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT(nn.Module):
    def __init__(self, num_emotions, hidden_dim):
        super(BERT, self).__init__()
        
        custom_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=hidden_dim,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            vocab_size=30522
        )
        
        self.bert = BertModel(config=custom_config)
        
        self.fc = nn.Linear(custom_config.hidden_size, num_emotions)
    
    def forward(self, x): #(b,t,hidden_dim)
        
        bert_output = self.bert(inputs_embeds=x)
        
        last_hidden_state = bert_output.last_hidden_state  #(b,t,hidden_dim)
        
        output = self.fc(last_hidden_state)  # Shape: (b,t,e)
        
        return output
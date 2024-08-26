import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT(nn.Module):
    def __init__(self,
                 num_emotions: int,
                 input_dim: int,
                 seq_len: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 intermediate_size: int = 2048,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 vocab_size: int = 30522,
                 pretrained_model_name: str = None
                 ) -> None:
        super(BERT, self).__init__()

        if pretrained_model_name:
            print(f"Loading pretrained model {pretrained_model_name}")
            self.bert = BertModel.from_pretrained(pretrained_model_name)
            custom_config = self.bert.config
        else:
            custom_config = BertConfig(
                hidden_size=input_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=seq_len,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
                vocab_size=vocab_size
            )
            self.bert = BertModel(config=custom_config)
            
        print(self.bert.config)

        if input_dim != custom_config.hidden_size:
            self.input_projection = nn.Linear(input_dim, custom_config.hidden_size)
        else:
            self.input_projection = nn.Identity()  # no transformation needed

        self.fc = nn.Linear(custom_config.hidden_size, num_emotions)

    def forward(self, x):
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)

        x = self.input_projection(x)

        bert_output = self.bert(inputs_embeds=x)

        last_hidden_state = bert_output.last_hidden_state

        output = self.fc(last_hidden_state)

        return output

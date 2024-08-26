import torch.nn as nn
from transformers import BertModel, BertConfig

class BERT(nn.Module):
    def __init__(self,
                 num_emotions: int,
                 input_dim: int,    # Dimension of your custom input embeddings
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
            # Load the pretrained BERT model
            self.bert = BertModel.from_pretrained(pretrained_model_name)
            # Use the config from the pretrained model
            custom_config = self.bert.config
        else:
            # Initialize with custom configuration
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
            
        # Print the configuration of the BERT model
        print(self.bert.config)

        # If input_dim does not match hidden_size, add a linear layer to match the dimensions
        if input_dim != custom_config.hidden_size:
            self.input_projection = nn.Linear(input_dim, custom_config.hidden_size)
        else:
            self.input_projection = nn.Identity()  # No transformation needed

        self.fc = nn.Linear(custom_config.hidden_size, num_emotions)

    def forward(self, x):
        # Reshape and transpose input as before
        x = x.flatten(start_dim=1, end_dim=4).transpose(1, 2)

        # Project input to the size expected by BERT if necessary
        x = self.input_projection(x)

        # Pass through the BERT model
        bert_output = self.bert(inputs_embeds=x)

        # Get the last hidden state
        last_hidden_state = bert_output.last_hidden_state

        # Pass the last hidden state through the fully connected layer
        output = self.fc(last_hidden_state)

        return output

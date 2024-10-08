from .SwiFT.swin4d_transformer_ver7 import SwinTransformer4D
from .bert import BERT
from .mlp import SimpleMLP
from .linear import LinearLayer
from .lstm import LSTM

def load_model(model_name, hparams=None):
    #number of transformer stages
    #n_stages = len(hparams.depths) -> assume four stages

    if hparams.precision == 16:
        to_float = False
    elif hparams.precision == 32:
        to_float = True

    print(to_float)
    
    h, w, d, t = hparams.img_size
    dims = h//48 * w//48 * d//48 * hparams.embed_dim*8 # TODO: verify this
    
    if model_name == "swin4d_ver7":
        net = SwinTransformer4D(
            img_size=hparams.img_size,
            in_chans=hparams.in_chans,
            embed_dim=hparams.embed_dim,
            window_size=hparams.window_size,
            first_window_size=hparams.first_window_size,
            patch_size=hparams.patch_size,
            depths=hparams.depths,
            num_heads=hparams.num_heads,
            c_multiplier=hparams.c_multiplier,
            last_layer_full_MSA=hparams.last_layer_full_MSA,
            to_float = to_float,
            drop_rate=hparams.attn_drop_rate,
            drop_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate
        )
    elif model_name == "bert":
        net = BERT(hparams.target_dim,
                dims,
                t,
                pretrained_model_name=hparams.bert_pretrained_model_name,
                num_layers=hparams.bert_num_layers,
                num_heads=hparams.bert_num_heads,
                intermediate_size=hparams.bert_intermediate_size,
                hidden_dropout_prob=hparams.bert_hidden_dropout_prob,
                attention_probs_dropout_prob=hparams.bert_attention_probs_dropout_prob,
                vocab_size=hparams.bert_vocab_size)
    elif model_name == "mlp":
        net = SimpleMLP(dims, hparams.mlp_dim, hparams.target_dim)
    elif model_name == "linear":
        net = LinearLayer(dims, hparams.target_dim)
    elif model_name == "lstm":
        net = LSTM(dims, hparams.lstm_dim, hparams.lstm_layers, hparams.target_dim)
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net

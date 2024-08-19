from .swiftbert import SwiFTBERT

def load_model(model_name, hparams=None):
    #number of transformer stages
    #n_stages = len(hparams.depths) -> assume four stages

    if hparams.precision == 16:
        to_float = False
    elif hparams.precision == 32:
        to_float = True

    print(to_float)

    if model_name.lower() == "swiftbert":
        net = SwiFTBERT(
            img_size=hparams.img_size,
            in_channels=hparams.in_chans,
            out_channels=hparams.out_chans,
            target_dim=hparams.target_dim,
            mlp_dim=hparams.mlp_dim,
            patch_size=hparams.patch_size,
            depths=hparams.depths,
            window_size=hparams.window_size,
            first_window_size=hparams.first_window_size,
            drop_rate=hparams.attn_drop_rate,
            dropout_path_rate=hparams.attn_drop_rate,
            attn_drop_rate=hparams.attn_drop_rate,
            feature_size=hparams.embed_dim,
            num_heads=hparams.num_heads,
            last_layer_full_MSA=hparams.last_layer_full_MSA,
            to_float=to_float,
            )
    else:
        raise NameError(f"{model_name} is a wrong model name")

    return net

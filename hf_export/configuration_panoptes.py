# coding=utf-8
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class PanoptesConfig(PretrainedConfig):
    model_type = "panoptes"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=100257,
        d_model=384,
        n_layers=6,
        n_heads=6,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.0,
        norm="pre",
        tie_embeddings=True,
        mtp_depth=4,
        # PANOPTES specific configuration dictionary
        panoptes_cfg=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.norm = norm
        self.tie_embeddings = tie_embeddings
        self.mtp_depth = mtp_depth
        
        # Fallback empty dict if no config passed
        self.panoptes_cfg = panoptes_cfg if panoptes_cfg is not None else {}
        
        # Pull out model config specifically for easy access
        if "model" not in self.panoptes_cfg:
            self.panoptes_cfg["model"] = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "dropout": dropout,
                "norm": norm,
                "tie_embeddings": tie_embeddings,
            }
        
        super().__init__(**kwargs)


from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class CortexConfig(PretrainedConfig):
    model_type = "cortex"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=100257,
        d_model=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=None,
        d_ff=1536,
        max_seq_len=512,
        rope_theta=500000.0,
        sliding_window=None,
        quantize_kv=False,
        dropout=0.0,
        norm="pre",
        tie_embeddings=True,
        mtp_depth=4,
        
        cortex_cfg=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.quantize_kv = quantize_kv
        self.dropout = dropout
        self.norm = norm
        self.tie_embeddings = tie_embeddings
        self.mtp_depth = mtp_depth
        
        
        self.cortex_cfg = cortex_cfg if cortex_cfg is not None else {}
        
        
        if "model" not in self.cortex_cfg:
            self.cortex_cfg["model"] = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "n_kv_heads": self.n_kv_heads,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "rope_theta": self.rope_theta,
                "sliding_window": self.sliding_window,
                "quantize_kv": self.quantize_kv,
                "dropout": dropout,
                "norm": norm,
                "tie_embeddings": tie_embeddings,
            }
        
        super().__init__(**kwargs)

import torch
import torch.nn as nn
from typing import List, Optional

class MockPagedAttention(nn.Module):

    def forward(self, q, k, v, kv_cache, attn_metadata, g_t: Optional[torch.Tensor]=None):
        pass

class MockLatentPonderKernel(nn.Module):

    def forward(self, hidden_states: torch.Tensor, g_t: torch.Tensor, max_steps: int=4):
        pass

class CortexAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = MockPagedAttention()

    def forward(self, hidden_states, kv_cache, attn_metadata, g_t):
        attn_output = self.attn(q=None, k=None, v=None, kv_cache=kv_cache, attn_metadata=attn_metadata, g_t=g_t)
        return attn_output

class CortexDecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attn = CortexAttention(config)

    def forward(self, hidden_states, kv_cache, attn_metadata, g_t):
        attn_out = self.self_attn(hidden_states, kv_cache, attn_metadata, g_t)
        return hidden_states

class CortexModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.controller = nn.Linear(config.d_model, 1) if config.cortex_cfg['subsystems']['controller'] else None
        self.layers = nn.ModuleList([CortexDecoderLayer(config) for _ in range(config.n_layers)])
        self.latent_ponder = MockLatentPonderKernel() if config.cortex_cfg['subsystems']['latent_loop'] else None

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden_states = torch.randn(1, 1, self.config.d_model)
        g_t = None
        if self.controller is not None:
            g_t = torch.sigmoid(self.controller(hidden_states))
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, kv_caches[i] if kv_caches is not None else None, attn_metadata, g_t=g_t)
        if self.latent_ponder is not None:
            hidden_states = self.latent_ponder(hidden_states, g_t)
        return hidden_states

class CortexForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = CortexModel(config)

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return hidden_states
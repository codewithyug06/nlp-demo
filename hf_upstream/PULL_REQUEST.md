# [Model] Add CORTEX (Adaptive Compute Transformer)

## PR Type
- [x] New Model Architecture

## What does this PR do?

This PR introduces the **CORTEX (Controller Orchestrated Routing Transformer with Expert eXecution)** architecture to the Hugging Face `transformers` library.

CORTEX is a massive leap forward in inference efficiency. Instead of allocating the same number of FLOPs to every token (like a standard dense LLM), CORTEX uses a **Universal Compute Controller** ($g_t$) to dynamically route compute based on the difficulty of the token.

### Key Innovations:
1. **Dynamic Byte-Patching**: Easy tokens skip embedding lookup and are patched instantly.
2. **MoE Imbalance Routing**: The controller forces experts to load-balance based on compute difficulty.
3. **Latent Ponder Loop**: Hard reasoning tokens (math, code) are looped through the block repeatedly before exiting.
4. **Attention Budgeting**: The KV cache is selectively pruned based on the controller's estimation of token importance.
5. **Vision Integration**: Natively supports visual inputs for multi-modal dynamic compute routing.

## Model Mapping:
- `CortexConfig` in `src/transformers/models/cortex/configuration_cortex.py`
- `CortexForCausalLM` in `src/transformers/models/cortex/modeling_cortex.py`

## Testing
We have included full baseline parity tests ensuring that when `enable_controller=False`, CORTEX converges identically to a standard dense LLaMA-style decoder.

## Who can review?
@ArthurZucker @younesbelkada @sgugger 

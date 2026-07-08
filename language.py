"""Real-world language modeling data loader (Phase 3.1).

Replaces synthetic integer tasks with real text streaming, BPE tokenization,
and proper packing for training a production-grade LLM.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Iterator

@dataclass
class LanguageBatch:
    x: torch.Tensor             # (B, T) input tokens
    y: torch.Tensor             # (B, T) target tokens (shifted)
    difficulty: torch.Tensor    # (B, T) Ground truth difficulty (optional, 1.0 for loss mask)

class LanguageDataset:
    """Streams tokenized text from a HuggingFace dataset."""
    
    def __init__(self, dataset_name: str, split: str = "train", seq_len: int = 512, batch_size: int = 32, device="cpu"):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        
        try:
            from datasets import load_dataset
            import tiktoken
        except ImportError:
            raise ImportError("Please install `datasets` and `tiktoken` to use the LanguageDataset.")
            
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
        self.enc = tiktoken.get_encoding("cl100k_base") # Standard OpenAI BPE
        self.vocab_size = self.enc.n_vocab
        self.iterator = iter(self.dataset)
        self.token_buffer = []

    def _fill_buffer(self, min_tokens: int):
        """Pulls documents from the stream until we have enough tokens."""
        while len(self.token_buffer) < min_tokens:
            try:
                row = next(self.iterator)
                text = row.get("text", row.get("content", ""))
                if not text.strip():
                    continue
                tokens = self.enc.encode_ordinary(text)
                # Append EOS token (using 100257 as standard cl100k EOS)
                tokens.append(100257)
                self.token_buffer.extend(tokens)
            except StopIteration:
                # Loop dataset if exhausted
                self.iterator = iter(self.dataset)

    def __iter__(self) -> Iterator[LanguageBatch]:
        return self
        
    def __next__(self) -> LanguageBatch:
        tokens_needed = self.batch_size * (self.seq_len + 1)
        self._fill_buffer(tokens_needed)
        
        # Take exact chunk needed for the batch
        chunk = self.token_buffer[:tokens_needed]
        self.token_buffer = self.token_buffer[tokens_needed:]
        
        # Shape into (B, T+1)
        tensor = torch.tensor(chunk, dtype=torch.long, device=self.device)
        tensor = tensor.view(self.batch_size, self.seq_len + 1)
        
        x = tensor[:, :-1]
        y = tensor[:, 1:]
        
        # In standard unsupervised LM, all tokens are equally weighted in the loss
        difficulty = torch.ones_like(x, dtype=torch.float32)
        
        return LanguageBatch(x=x, y=y, difficulty=difficulty)

def make_language_batch(
    dataset_name: str,
    batch_size: int,
    seq_len: int,
    device: torch.device | str = "cpu",
) -> LanguageBatch:
    """Helper to fetch a single batch (mostly for testing)."""
    ds = LanguageDataset(dataset_name, seq_len=seq_len, batch_size=batch_size, device=device)
    return next(ds)

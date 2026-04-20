import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, hidden_size=64, num_heads=4, ff_size=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Linear(ff_size, hidden_size)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Attention block
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward block
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

def get_model():
    # Hidden size 512, 12 heads is more realistic but tiny is good for testing
    model = TinyTransformer(hidden_size=256, num_heads=8, ff_size=1024)
    # (Batch, SeqLen, Hidden)
    example_inputs = (torch.randn(8, 512, 256),)
    return model, example_inputs

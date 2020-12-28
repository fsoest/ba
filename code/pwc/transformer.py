import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_encoder, n_decoder, d_ff):
        super(Transformer, self).__init__()
        

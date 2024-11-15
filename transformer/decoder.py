import torch
import torch.nn as nn

from .attention import MultiHeadAttention, FeedForwardNN

class DecoderLayer(nn.Module):
    
    def __init__(self, 
                 num_heads: int, 
                 ffn_hidden_dim: int,
                 qk_length: int, 
                 value_length: int):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """

        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        raise NotImplementedError("Implement the EncoderLayer layer definitions!")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the EncoderLayer.
        """
        raise NotImplementedError("Implement the EncoderLayer forward method!")


class Decoder(nn.Module):

    def __init__(self, 
                 vocab_size: int, 
                 num_layers: int, 
                 num_heads: int,
                 ffn_hidden_dim: int,
                 qk_length: int,
                 value_length: int):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers 
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        # 
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.
        raise NotImplementedError("Implement the Decoder layer definitions!")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        raise NotImplementedError("Implement the Encoder forward method!")

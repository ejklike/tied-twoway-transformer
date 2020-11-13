"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.mean_encoder import MeanEncoder


str2enc = {"transformer": TransformerEncoder,
           "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "MeanEncoder", "str2enc"]

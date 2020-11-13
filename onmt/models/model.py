""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder_x2y, encoder_y2x, decoder_x2y, decoder_y2x):
        super(NMTModel, self).__init__()
        self.encoder_x2y = encoder_x2y
        self.encoder_y2x = encoder_y2x
        self.decoder_x2y = decoder_x2y
        self.decoder_y2x = decoder_y2x

    def forward(self, src, dec_in, lengths, with_align=False, side='x2y'):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        encoder = self.encoder_x2y if side == 'x2y' else self.encoder_y2x
        decoder = self.decoder_x2y if side == 'x2y' else self.decoder_y2x

        enc_state, memory_bank, lengths = encoder(src, lengths)
        
        decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = decoder(dec_in, memory_bank,
                                 memory_lengths=lengths,
                                 with_align=with_align)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder_x2y.update_dropout(dropout)
        self.encoder_y2x.update_dropout(dropout)
        self.decoder_x2y.update_dropout(dropout)
        self.decoder_y2x.update_dropout(dropout)

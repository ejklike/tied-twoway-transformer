"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders import str2enc

from onmt.decoders import str2dec

from onmt.modules import Embeddings, VecEmbedding, CopyGenerator
from onmt.modules.util_class import Cast
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
from onmt.utils.parse import ArgumentParser


def build_embeddings(opt, text_field):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    assert opt.src_word_vec_size == opt.tgt_word_vec_size
    emb_dim = opt.src_word_vec_size

    pad_indices = [f.vocab.stoi[f.pad_token] for _, f in text_field]
    word_padding_idx, feat_pad_indices = pad_indices[0], pad_indices[1:]

    num_embs = [len(f.vocab) for _, f in text_field]
    num_word_embeddings, num_feat_embeddings = num_embs[0], num_embs[1:]

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam"
    )
    return emb


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    enc_type = opt.encoder_type if opt.model_type == "text" \
        or opt.model_type == "vec" else opt.model_type
    return str2enc[enc_type].from_opt(opt, embeddings)


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
               else opt.decoder_type
    return str2dec[dec_type].from_opt(opt, embeddings)


def load_test_model(opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    model_opt = ArgumentParser.ckpt_model_opts(checkpoint['opt'])
    ArgumentParser.update_model_opts(model_opt)
    ArgumentParser.validate_model_opts(model_opt)
    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_old_vocab(
            vocab, opt.data_type, dynamic_dict=model_opt.copy_attn
        )
    else:
        fields = vocab

    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint,
                             opt.gpu)
    if opt.fp32:
        model.float()
    model.eval()
    model.generator_x2y.eval()
    model.generator_y2x.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
    """Build a model from opts.

    Args:
        model_opt: the option loaded from checkpoint. It's important that
            the opts have been updated and validated. See
            :class:`onmt.utils.parse.ArgumentParser`.
        fields (dict[str, torchtext.data.Field]):
            `Field` objects for the model.
        gpu (bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
        gpu_id (int or NoneType): Which GPU to use.

    Returns:
        the NMTModel.
    """

    # for back compat when attention_dropout was not defined
    try:
        model_opt.attention_dropout
    except AttributeError:
        model_opt.attention_dropout = model_opt.dropout

    # Build embeddings.
    src_field = fields["src"]
    src_emb = build_embeddings(model_opt, src_field)
    tgt_field = fields["tgt"]
    tgt_emb = build_embeddings(model_opt, tgt_field)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    # Build encoder.
    encoder_x2y = build_encoder(model_opt, src_emb)
    encoder_y2x = build_encoder(model_opt, tgt_emb)

    # Build decoder.
    decoder_x2y = build_decoder(model_opt, tgt_emb)
    decoder_y2x = build_decoder(model_opt, src_emb)


    def share_attn_weight_and_bias(attn1, attn2, 
                                   share_relative_pos_embeddings=False):
        attn2.linear_keys = attn1.linear_keys
        attn2.linear_values = attn1.linear_values
        attn2.linear_query = attn1.linear_query
        attn2.final_linear = attn1.final_linear
        if share_relative_pos_embeddings:
            assert model_opt.max_relative_positions > 0
            attn2.relative_positions_embeddings = \
                attn1.relative_positions_embeddings

    # logger.info('share encoder')
    encoder_y2x = encoder_x2y
    # logger.info('share cross_attns btw fwd & bwd decoders')
    for dec1, dec2 in zip(decoder_x2y.transformer_layers, 
                            decoder_y2x.transformer_layers):
        share_attn_weight_and_bias(dec1.context_attn, dec2.context_attn)

    # logger.info('share self_attns btw fwd & bwd decoders')
    for dec1, dec2 in zip(decoder_x2y.transformer_layers, 
                            decoder_y2x.transformer_layers):
        share_attn_weight_and_bias(dec1.self_attn, dec2.self_attn,
                                    model_opt.share_relative_pos_embeddings)
    # logger.info('share feed_forwards btw fwd & bwd decoders')
    for dec1, dec2 in zip(decoder_x2y.transformer_layers, 
                            decoder_y2x.transformer_layers):
        dec2.feed_forward.w_1 = dec1.feed_forward.w_1
        dec2.feed_forward.w_2 = dec1.feed_forward.w_2

    # Build NMTModel(= encoder + decoder).
    if gpu and gpu_id is not None:
        device = torch.device("cuda", gpu_id)
    elif gpu and not gpu_id:
        device = torch.device("cuda")
    elif not gpu:
        device = torch.device("cpu")
    model = onmt.models.NMTModel(encoder_x2y, encoder_y2x, 
                                 decoder_x2y, decoder_y2x)

    # Build prior modeling
    prior = None
    if model_opt.learned_prior:
        assert model_opt.num_experts > 1
        prior = onmt.models.Classifier(
            model_opt.enc_rnn_size, model_opt.num_experts, 
            dropout=(model_opt.dropout[0] if type(model_opt.dropout) is list
                     else model_opt.dropout))

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator_x2y = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["tgt"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        generator_y2x = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size,
                      len(fields["src"].base_field.vocab)),
            Cast(torch.float32),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator_x2y[0].weight = decoder_x2y.embeddings.word_lut.weight
            generator_y2x[0].weight = decoder_y2x.embeddings.word_lut.weight
    else:
        tgt_base_field = fields["tgt"].base_field
        vocab_size = len(tgt_base_field.vocab)
        pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)
        if model_opt.share_decoder_embeddings:
            generator_x2y.linear.weight = decoder_x2y.embeddings.word_lut.weight
            generator_y2x.linear.weight = decoder_y2x.embeddings.word_lut.weight

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator_x2y.load_state_dict(checkpoint['generator_x2y'], strict=False)
        generator_y2x.load_state_dict(checkpoint['generator_y2x'], strict=False)
        if model_opt.learned_prior:
            prior.load_state_dict(checkpoint['prior'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            def init_param(target_model):
                for p in target_model.parameters():
                    p.data.uniform_(-model_opt.param_init, 
                                    model_opt.param_init)
            init_param(model)
            init_param(generator_x2y)
            init_param(generator_y2x)
            if model_opt.learned_prior:
                init_param(prior)
        if model_opt.param_init_glorot:
            def init_glorot(target_model):
                for p in target_model.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
            init_glorot(model)
            init_glorot(generator_x2y)
            init_glorot(generator_y2x)
            if model_opt.learned_prior:
                init_glorot(prior)

    model.generator_x2y = generator_x2y
    model.generator_y2x = generator_y2x
    model.prior = prior
    model.to(device)
    if model_opt.model_dtype == 'fp16' and model_opt.optim == 'fusedadam':
        model.half()
    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    # logger.info(model)
    return model

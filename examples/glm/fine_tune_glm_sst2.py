

# -*- encoding: utf-8 -*-
'''
@File    :   finetune_glm_sst2.py
@Time    :   2021/12/12 20:53:28
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

from SwissArmyTransformer.data_utils.datasets import TSVDataset
from SwissArmyTransformer.model.cached_autoregressive_model import CachedAutoregressiveMixin
import torch
import argparse
import numpy as np

from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model.base_model import AutoModel, BaseModel, BaseMixin, non_conflict
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import TSVDataset
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.model.transformer import standard_attention
from SwissArmyTransformer.model.mixins import MLPHeadMixin, PrefixTuningMixin

class GenerationModel(GLMModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('classification_head',
                       MLPHeadMixin(args.hidden_size, 2048, 1))
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size //
                       args.num_attention_heads, args.num_attention_heads, args.prefix_len))

    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['sentence', 'label']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['sentence'].long()
    labels = data_b['label'].long()
    batch_size, seq_length = tokens.size()

    position_ids = torch.zeros(
        2, seq_length, device=tokens.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[0, :seq_length])
    position_ids = position_ids.unsqueeze(0)

    attention_mask = torch.ones(
        (batch_size, 1, seq_length, seq_length), device=tokens.device)

    attention_mask[..., :seq_length] -= (tokens == -
                                         1).view(batch_size, 1, 1, seq_length).float()
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    return tokens, labels, attention_mask, position_ids, (tokens != -1)


def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, attention_mask, position_ids, loss_mask = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask)
    pred = ((logits.contiguous().float().squeeze(-1)) *
            loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        labels.float()
    )
    acc = ((pred > 0.).long() == labels).sum() / labels.numel()
    return loss, {'acc': acc}


def create_dataset_function(path, args):
    tokenizer = get_tokenizer()

    def process_fn(row):
        sentence, label = tokenizer._encode(row[0]), int(row[1])
        sentence = [tokenizer.get_command(
            'ENC').Id] + sentence + [tokenizer.get_command('eos').Id]
        if len(sentence) >= args.sample_length:
            sentence = sentence[:args.sample_length]
        else:
            sentence.extend([-1] * (args.sample_length-len(sentence)))
        return {'sentence': np.array(sentence, dtype=np.int64), 'label': label}
    return TSVDataset(path, process_fn, with_heads=True)


if __name__ == '__main__':
    model, args = AutoModel.from_pretrained(args, 'glm-10b-chinese')
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    # Generate a sequence with beam search
    from SwissArmyTransformer.generation.autoregressive_sampling import filling_sequence
    from SwissArmyTransformer.generation.sampling_strategies import BeamSearchStrategy
    output, *mems = filling_sequence(model, input_seq,
                                     batch_size=args.batch_size,
                                     strategy=BeamSearchStrategy(args.batch_size))

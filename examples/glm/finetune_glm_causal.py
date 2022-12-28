
# here put the import lib
from SwissArmyTransformer.data_utils import load_hf_dataset
import os
import sys
import math
import random
import gc 

from SwissArmyTransformer.data_utils.datasets import TSVDataset
import torch
import argparse
import numpy as np

from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model.base_model import BaseModel, BaseMixin, non_conflict
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import TSVDataset
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.model.transformer import standard_attention
from SwissArmyTransformer.model.mixins import MLPHeadMixin, PrefixTuningMixin, CachedAutoregressiveMixin

class AutoRegressiveModel(GLMModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))

    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        for layer_id in range(len(self.transformer.layers)):
            self.transformer.layers[layer_id].requires_grad_(False)

def get_masks(data, loss_mask=None, attention_mask=None, args=None):
    batch_size, seq_length = data.size()

    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length, seq_length), device=data.device)
        attention_mask.tril_()
        attention_mask.unsqueeze_(1)
        
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=data.dtype, device=data.device)

    return attention_mask, loss_mask


def get_batch(data_iterator, args, timers):
    keys = ['text', 'loss_mask']
    datatype = torch.int64

    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    data_b = mpu.broadcast_data(keys, data, datatype)
    tokens_ = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()
    
    labels = tokens_[:, 1:].contiguous()
    loss_mask = loss_mask[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    _, seq_length = tokens.size()

    position_ids = torch.zeros(2, seq_length, device=tokens.device, dtype=torch.long)
    torch.arange(0, seq_length, out=position_ids[0, :seq_length])
    position_ids = position_ids.unsqueeze(0)

    attention_mask = None        
    attention_mask, loss_mask = get_masks(
        tokens,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        args=args
        )

    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids 

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()
    
    logits, *mems = model(tokens, position_ids, attention_mask)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)

    loss_mask = loss_mask.view(-1)  
    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / torch.sum(loss_mask)
    return loss, {}

def create_dataset_function(path, args):
    tokenizer = get_tokenizer(args)
    
    def process_fn(row):
        dialog = row['dialog']
        prefix = dialog[0]
        text = ''.join(dialog)  
        sentence = tokenizer._encode(text[:])
        sentence = sentence + [tokenizer.get_command('eos').Id]
        # TODO: Re-evaluate: No loss on the first sentence
        loss_mask = [0]*len(prefix) + [1] * (args.sample_length - len(prefix))
        if len(sentence) >= args.sample_length:
            # Cutoff on rhs
            sentence = sentence[:args.sample_length]
        else:
            n_pad = args.sample_length-len(sentence)
            loss_mask = np.array([1]*len(sentence) + [0]*n_pad)
            sentence.extend([tokenizer.get_command('pad').Id] * n_pad)

        return {'text': np.array(sentence, dtype=np.int64), 'loss_mask': np.array(loss_mask, dtype=np.int64)}
    return load_hf_dataset(path, process_fn, columns=['text', 'loss_mask'], cache_dir='~/dataset/SwissArmyTransformerDatasets', offline=False)

if __name__ == '__main__':   
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--new_hyperparam', type=str, default=None)
    py_parser.add_argument('--sample_length', type=int, default=1024-64)
    py_parser.add_argument('--prefix_len', type=int, default=64)
    GLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    training_main(args, model_cls=AutoRegressiveModel, forward_step_function=forward_step, create_dataset_function=create_dataset_function)

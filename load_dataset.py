import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from build_tokenizer import build_tokenizer

from bilingual_dataset import BilingualDataset, casual_mask


def load_ds(config):
    dataset_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_target"]}', split='train')

    # build tokenizers
    tokenizer_src = build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_target = build_tokenizer(config, dataset_raw, config['lang_target'])

    # split to train and validation ds
    train_dataset_size = int(0.9 * len(dataset_raw))
    validation_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, validation_dataset_raw = random_split(dataset_raw, [train_dataset_size, validation_dataset_size])

    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_target, config['lang_src'],
                                     config['lang_target'], config['seq_len'])

    validation_dataset = BilingualDataset(validation_dataset_raw, tokenizer_src, tokenizer_target, config['lang_src'],
                                          config['lang_target'], config['seq_len'])

    max_len_src = 0
    max_len_target = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_src.encode(item['translation'][config['lang_target']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_target = max(max_len_target, len(target_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_target}')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    return train_dataloader, validation_dataloader, tokenizer_src, tokenizer_target

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_target, src_lang, target_lang, seq_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_target = tokenizer_target
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len
        # Define special tokens
        special_tokens = ['SOS', 'EOS', 'PAD']

        # Add special tokens if they are not already present
        tokenizer_src.add_special_tokens(special_tokens)

        # Get the IDs of special tokens
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('SOS')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('EOS')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('PAD')], dtype=torch.int64)
        # Add any other initializations you need

        # Check if tokens were added correctly
        assert self.sos_token.item() is not None, "SOS token was not added correctly."
        assert self.eos_token.item() is not None, "EOS token was not added correctly."
        assert self.pad_token.item() is not None, "PAD token was not added correctly."

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids
        decoder_input_tokens = self.tokenizer_target.encode(target_text).ids

        encoder_num_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2
        decoder_num_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # add EOS token to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # add EOS to the  label (what we except as output from decoder)
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & casual_mask(
                decoder_input.size(0)),  # (1,seq_len) & (1,seq_len,seq_len)
            "src_text": src_text,
            "label": label,
            "target_text": target_text
        }


def casual_mask(size):
    # hide masked the decoder input upper diagonal to restrict the model to see future words
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

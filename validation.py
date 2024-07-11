import torch
import torch.nn as nn
from bilingual_dataset import casual_mask
def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_target, max_len, device):
    sos_inx = tokenizer_target.token_to_id('[SOS]')
    eos_inx = tokenizer_target.token_to_id('[EOS]')

    # Precompute encoder output and reuse i for every token we get from the decoder
    encoder_output = model.encode(src, src_mask)

    # init decoder input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_inx).type_as(src).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for the decoder input
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(src_mask).to(device)
        out = model.decode(encoder_output, src_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:, -1])
        # select the token with max prob (greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(src).fill_(next_word.item()).to(device)],
                                  dim=1)
        if next_word == eos_inx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_dataset, tokenizer_src, tokenizer_target, max_len, device, print_msg, global_state,
                   writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    # size of the control windows(just use a default value)
    console_width = 80
    with torch.no_grad():
        for batch in validation_dataset:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_target, max_len,
                                      device)

            source_text = batch['src_text'][0]
            target_text = batch['target_text'][0]
            model_out_text = tokenizer_target.decode(model_out.detach().cpu().numpy())

            print_msg('-' * console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break
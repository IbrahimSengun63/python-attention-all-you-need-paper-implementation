from build_transformer import build_transformer


def build_model(config, vocab_src_len, vocab_target_len):
    model = build_transformer(vocab_src_len, vocab_target_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

import torch


def contains_digit(string):
    return any(char.isdigit() for char in string)


def tokens_to_seq(tokens, tok_to_idx, max_length, use_extended_vocab, input_tokens=None):
    seq = torch.zeros(max_length).long()
    tok_to_idx_extension = dict()

    for pos, token in enumerate(tokens):
        if token in tok_to_idx:
            idx = tok_to_idx[token]

        elif token in tok_to_idx_extension:
            idx = tok_to_idx_extension[token]

        elif use_extended_vocab and input_tokens is not None:
            # If the token is not in the vocab and an input token sequence was provided
            # find the position of the first occurance of the token in the input sequence
            # the token index in the output sequence is size of the vocab plus the position in the input sequence.
            # If the token cannot be found in the input sequence use the unknown token.

            tok_to_idx_extension[token] = tok_to_idx_extension.get(token,
                                                                   next((pos + len(tok_to_idx)
                                                                         for pos, input_token in enumerate(input_tokens)
                                                                         if input_token == token), 3))
            idx = tok_to_idx_extension[token]

        elif use_extended_vocab:
            # unknown tokens in the input sequence use the position of the first occurence + vocab_size as their index
            idx = pos + len(tok_to_idx)
        else:
            idx = tok_to_idx['<UNK>']

        seq[pos] = idx

    return seq

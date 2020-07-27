from models.util import seq_to_string, tokens_to_seq
import torch
from torch import nn
from models.rnn_encoder import RNNEncoder
from models.biattention_encoder import BiAttentionEncoder
from models.attention_decoder import AttentionDecoder
from models.copy_decoder import CopyDecoder


class EncoderDecoder(nn.Module):
    def __init__(self, lang, max_tweet_length, max_news_length, max_hashtag_length, hidden_size, embedding_size,
                 encoder_type, decoder_type, decode_strategy, beam_width, tweet_cov_loss_factor=0,
                 news_cov_loss_factor=0):
        super(EncoderDecoder, self).__init__()

        self.lang = lang
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        if self.encoder_type == 'rnn':
            self.encoder = RNNEncoder(input_size=len(self.lang.tok_to_idx),
                                      hidden_size=hidden_size,
                                      embedding_size=embedding_size)
        elif self.encoder_type == "biattention":
            self.encoder = BiAttentionEncoder(input_size=len(self.lang.tok_to_idx),
                                              hidden_size=hidden_size,
                                              embedding_size=embedding_size)
        else:
            raise ValueError("decoder_type must be 'rnn' or 'biattention'")

        decoder_hidden_size = 2 * self.encoder.hidden_size

        if self.decoder_type == 'attn':
            self.decoder = AttentionDecoder(hidden_size=decoder_hidden_size,
                                            embedding_size=embedding_size,
                                            lang=lang,
                                            max_hashtag_length=max_hashtag_length,
                                            decode_strategy=decode_strategy,
                                            beam_width=beam_width)
        elif self.decoder_type == 'copy':
            self.decoder = CopyDecoder(hidden_size=decoder_hidden_size,
                                       embedding_size=embedding_size,
                                       lang=lang,
                                       max_tweet_length=max_tweet_length,
                                       max_news_length=max_news_length,
                                       max_hashtag_length=max_hashtag_length,
                                       decode_strategy=decode_strategy,
                                       beam_width=beam_width,
                                       tweet_cov_loss_factor=tweet_cov_loss_factor,
                                       news_cov_loss_factor=news_cov_loss_factor)
        else:
            raise ValueError("decoder_type must be 'attn' or 'copy'")

    def forward(self, input_tweets, input_news, lengths_tweets, lengths_news, targets=None, keep_prob=1.0,
                teacher_forcing=0.0):
        encoder_outputs, hidden = self.encoder(input_tweets,
                                               input_news,
                                               lengths_tweets,
                                               lengths_news)

        decoder_outputs, sampled_idxs, cov_loss = self.decoder(encoder_outputs,
                                                               input_tweets,
                                                               input_news,
                                                               hidden,
                                                               lengths_tweets=lengths_tweets,
                                                               lengths_news=lengths_news,
                                                               targets=targets,
                                                               teacher_forcing=teacher_forcing)
        return decoder_outputs, sampled_idxs, cov_loss

    def get_response(self, tweet_string, news_string):
        use_extended_vocab = not isinstance(self.decoder, AttentionDecoder)

        idx_to_tok = self.lang.idx_to_tok
        tok_to_idx = self.lang.tok_to_idx

        tweet_tokens = tweet_string.strip().split()
        tweet_tokens = ['<SOS>'] + [token.lower() for token in tweet_tokens] + ['<EOS>']
        input_seq = tokens_to_seq(tweet_tokens, tok_to_idx, len(tweet_tokens), use_extended_vocab)
        input_variable = input_seq.view(1, -1)

        news_tokens = news_string.strip().split()
        news_tokens = ['<SOS>'] + [token.lower() for token in news_tokens] + ['<EOS>']
        news_seq = tokens_to_seq(news_tokens, tok_to_idx, len(news_tokens), use_extended_vocab)
        news_variable = news_seq.view(1, -1)

        if next(self.parameters()).is_cuda:
            input_variable = input_variable.cuda()
            news_variable = news_variable.cuda()

        outputs, idxs, cov_loss = self.forward(input_variable, news_variable, torch.LongTensor([len(input_seq)]),
                                               torch.LongTensor([len(news_seq)]))
        idxs = idxs.data.view(-1)
        eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
        output_string = seq_to_string(idxs[:eos_idx + 1], idx_to_tok, input_tokens=tweet_tokens + news_tokens)

        return output_string

    def interactive(self, unsmear):
        while True:
            input_string = input("\nType a tweet:\n")
            news_string = input("\nType news corresponding to above tweet:\n")
            output_string = self.get_response(input_string, news_string)

            if unsmear:
                output_string = output_string.replace('<SOS>', '')
                output_string = output_string.replace('<EOS>', '')

            print('\nAmy:\n', output_string)

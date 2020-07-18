import argparse
import random

import torch

from dataset import TweetNewsDataset
from models.util import seq_to_string, to_np, trim_seqs
from models.encoder_decoder import EncoderDecoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def evaluate(encoder_decoder: EncoderDecoder, data_loader):
    loss_function = torch.nn.NLLLoss(ignore_index=0,
                                     reduce=False)  # what does this return for ignored idxs? same length output?

    losses = []
    all_output_seqs = []
    all_target_seqs = []

    for batch_idx, (tweet_idxs, news_idxs, target_idxs, _, _, _) in enumerate(tqdm(data_loader)):
        tweet_lengths = (tweet_idxs != 0).long().sum(dim=1)
        news_lengths = (news_idxs != 0).long().sum(dim=1)

        batch_size = tweet_idxs.shape[0]

        output_log_probs, output_seqs, cov_loss = encoder_decoder(tweet_idxs,
                                                                  news_idxs,
                                                                  tweet_lengths,
                                                                  news_lengths)

        all_output_seqs.extend(trim_seqs(output_seqs))
        all_target_seqs.extend([list(seq[seq > 0])] for seq in to_np(target_idxs))

        flattened_log_probs = output_log_probs.view(batch_size * encoder_decoder.decoder.max_length, -1)
        batch_losses = loss_function(flattened_log_probs, target_idxs.contiguous().view(-1))
        losses.extend(list(to_np(batch_losses)))

    mean_loss = len(losses) / sum(losses)

    bleu_score = corpus_bleu(all_target_seqs, all_output_seqs, smoothing_function=SmoothingFunction().method1)

    return mean_loss, bleu_score


def print_output(tweet_seq, news_seq, encoder_decoder: EncoderDecoder, input_tokens=None, news_tokens=None,
                 target_tokens=None, target_seq=None):
    idx_to_tok = encoder_decoder.lang.idx_to_tok

    if input_tokens is not None:
        input_tweets_string = ' '.join(input_tokens)
    else:
        input_tweets_string = seq_to_string(tweet_seq, idx_to_tok)

    if news_tokens is not None:
        input_news_string = ' '.join(news_tokens)
    else:
        input_news_string = seq_to_string(news_seq, idx_to_tok)

    lengths_tweets = list((tweet_seq != 0).long().sum(dim=0))
    lengths_news = list((news_seq != 0).long().sum(dim=0))

    input_tweets_variable = tweet_seq.view(1, -1)
    input_news_variable = news_seq.view(1, -1)
    target_variable = target_seq.view(1, -1)

    if target_tokens is not None:
        target_string = ' '.join(target_tokens)
    elif target_seq is not None:
        target_string = seq_to_string(target_seq, idx_to_tok, input_tokens=input_tokens)
    else:
        target_string = ''

    if target_seq is not None:
        target_eos_idx = list(target_seq).index(2) if 2 in list(target_seq) else len(target_seq)
        target_outputs, _ = encoder_decoder(input_tweets_variable,
                                            input_news_variable,
                                            lengths_tweets,
                                            lengths_news,
                                            targets=target_variable,
                                            teacher_forcing=0.0)
        target_log_prob = sum([target_outputs[0, step_idx, target_idx] for step_idx, target_idx in
                               enumerate(target_seq[:target_eos_idx + 1])])

    outputs, idxs = encoder_decoder(input_tweets_variable,
                                    input_news_variable,
                                    lengths_tweets,
                                    lengths_news)
    idxs = idxs.data.view(-1)
    eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
    string = seq_to_string(idxs[:eos_idx + 1], idx_to_tok, input_tokens=input_tokens)
    log_prob = sum([outputs[0, step_idx, idx] for step_idx, idx in enumerate(idxs[:eos_idx + 1])])

    print('>', input_tweets_string, '\n', flush=True)
    print('>', input_news_string, '\n', flush=True)

    if target_seq is not None:
        print('=', target_string, flush=True)
    print('<', string, flush=True)

    print('\n')

    if target_seq is not None:
        print('target log prob:', float(target_log_prob))
    print('output log prob:', float(log_prob))

    print('-' * 100, '\n')

    return idxs


def main(model_dump_path, test_dir, model_name, use_cuda, max_tweet_len, max_news_len, max_hashtag_len, n_print,
         val_size, batch_size, interact, unsmear):
    model_path = model_dump_path + model_name + '/'

    if use_cuda:
        encoder_decoder = torch.load(model_path + model_name + '.pt')
    else:
        encoder_decoder = torch.load(model_path + model_name + '.pt', map_location=lambda storage, loc: storage)

    if use_cuda:
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = encoder_decoder.cpu()

    dataset = TweetNewsDataset(data_dir=test_dir,
                               use_cuda=use_cuda,
                               lang=encoder_decoder.lang,
                               max_tweet_len=max_tweet_len,
                               max_news_len=max_news_len,
                               max_hashtag_len=max_hashtag_len,
                               use_extended_vocab=(encoder_decoder.decoder_type == 'copy'))

    data_loader = DataLoader(dataset, batch_size=batch_size)
    if interact:
        encoder_decoder.interactive(unsmear)

    if n_print is not None:
        for _ in range(n_print):
            tweet_seq, news_seq, target_seq, tweet_str, news_str, hashtag_str = random.choice(dataset)

            tweet_length = (tweet_seq > 0).sum()
            news_length = (news_seq > 0).sum()
            hashtag_length = (target_seq > 0).sum()

            i_seq = tweet_seq[:tweet_length]
            n_seq = news_seq[:news_length]
            t_seq = target_seq[:hashtag_length]

            i_tokens = tweet_str.strip().split()
            n_tokens = news_str.strip().split()
            t_tokens = news_str.strip().split()

            print_output(i_seq, n_seq, encoder_decoder, input_tokens=i_tokens, news_tokens=n_tokens,
                         target_tokens=t_tokens, target_seq=t_seq)

    else:
        evaluate(encoder_decoder, data_loader)


if __name__ == '__main__':
    random = random.Random(42)  # https://groups.google.com/forum/#!topic/nzpug/o4OW1O_4rgw

    arg_parser = argparse.ArgumentParser(description='Parse training parameters')

    arg_parser.add_argument('--model_dump_path', type=str,
                            help='The name of a directory of contains encoder and decoder model files.')

    arg_parser.add_argument('--model_name', type=str,
                            help='The name of a subdirectory of model_dump_path/ that '
                                 'contains encoder and decoder model files.')

    arg_parser.add_argument('--n_print', type=int,
                            help='Instead of evaluating the model on the entire dataset,'
                                 'n random examples from the dataset will be transformed.'
                                 'The output will be printed.')

    arg_parser.add_argument('--interact', action='store_true',
                            help='Take model inputs from the keyboard.')

    arg_parser.add_argument('--use_cuda', action='store_true',
                            help='A flag indicating that cuda will be used.')

    arg_parser.add_argument('--max_tweet_len', type=int, default=200,
                            help="Tweets will be padded or truncated to this size.")

    arg_parser.add_argument('--max_news_len', type=int, default=200,
                            help='News will be padded or truncated to this size.')

    arg_parser.add_argument('--max_hashtag_len', type=int, default=200,
                            help='Hashtag sequences will be padded or truncated to this size.')

    arg_parser.add_argument('--val_size', type=float, default=0.1,
                            help='The fractional size of the validation split.')

    arg_parser.add_argument('--batch_size', type=int, default=100,
                            help='The batch size to use when evaluating on the full dataset.')

    arg_parser.add_argument('--unsmear', action='store_true',
                            help='Replace <NUM> tokens with "1" and remove <SOS> and <EOS> tokens.')

    arg_parser.add_argument('--test_dir', type=str, default="./data/test",
                            help='test directory which contains tweets.txt, news.txt, hashtag.txt')

    args = arg_parser.parse_args()

    try:
        main(args.model_dump_path,
             args.test_dir,
             args.model_name,
             args.use_cuda,
             args.max_tweet_len,
             args.max_news_len,
             args.max_hashtag_len,
             args.n_print,
             args.val_size,
             args.batch_size,
             args.interact,
             args.unsmear)
    except KeyboardInterrupt:
        pass

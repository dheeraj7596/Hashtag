import argparse
import random

import torch

from dataset import TweetNewsDataset
from models.util import seq_to_string
from torch.utils.data import DataLoader
from tqdm import tqdm


def translate(encoder_decoder, test_data_loader, beam_width, n_best):
    assert beam_width >= n_best

    idx_to_tok = encoder_decoder.lang.idx_to_tok
    translations = []
    for batch_idx, (tweet_idxs, news_idxs, target_idxs, tweet_tokens, news_tokens, hashtag_tokens) in enumerate(
            tqdm(test_data_loader)):
        tweet_lengths = (tweet_idxs != 0).long().sum(dim=1)
        news_lengths = (news_idxs != 0).long().sum(dim=1)

        batch_size = tweet_idxs.shape[0]

        output_log_probs, output_seqs, cov_loss = encoder_decoder(tweet_idxs,
                                                                  news_idxs,
                                                                  tweet_lengths,
                                                                  news_lengths,
                                                                  beam_width=beam_width,
                                                                  n_best=n_best)
        output_seqs = output_seqs.squeeze(-1)  # (b_size x n_best x max_hashtag_length)

        for b_index in range(batch_size):
            tweet_tokens_list = tweet_tokens[b_index].split()
            news_tokens_list = news_tokens[b_index].split()
            n_best_translations = []
            for i in range(n_best):
                idxs = output_seqs[b_index, i, :]
                idxs = idxs.data.view(-1)
                eos_idx = list(idxs).index(2) if 2 in list(idxs) else len(idxs)
                output_string = seq_to_string(idxs[:eos_idx + 1],
                                              idx_to_tok,
                                              input_tokens=tweet_tokens_list + news_tokens_list)
                n_best_translations.append(output_string)

            translations.append(";".join(n_best_translations))
    return translations


def main(test_dir, model_path, use_cuda, max_tweet_len, max_news_len, max_hashtag_len, batch_size, out_file_path,
         beam_width, n_best):
    print("loading encoder and decoder from model_path", flush=True)
    if use_cuda:
        encoder_decoder = torch.load(model_path)
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = torch.load(model_path, map_location=torch.device('cpu'))
        encoder_decoder = encoder_decoder.cpu()

    print("creating test datasets with saved languages", flush=True)
    test_dataset = TweetNewsDataset(data_dir=test_dir,
                                    use_cuda=use_cuda,
                                    lang=encoder_decoder.lang,
                                    max_tweet_len=max_tweet_len,
                                    max_news_len=max_news_len,
                                    max_hashtag_len=max_hashtag_len,
                                    use_extended_vocab=(encoder_decoder.decoder_type == 'copy'))

    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    with torch.no_grad():
        translations = translate(encoder_decoder, test_data_loader, beam_width, n_best)

    with open(out_file_path, "w") as f:
        for t in translations:
            f.write(t)
            f.write("\n")


if __name__ == '__main__':
    random = random.Random(42)  # https://groups.google.com/forum/#!topic/nzpug/o4OW1O_4rgw

    arg_parser = argparse.ArgumentParser(description='Parse training parameters')

    arg_parser.add_argument('--test_dir', type=str, default="./data/test",
                            help='test directory which contains tweets.txt, news.txt, hashtag.txt')

    arg_parser.add_argument('--model_path', type=str,
                            help='The path of the model dump to use.')

    arg_parser.add_argument('--use_cuda', action='store_true',
                            help='A flag indicating that cuda will be used.')

    arg_parser.add_argument('--max_tweet_len', type=int, default=200,
                            help="Tweets will be padded or truncated to this size.")

    arg_parser.add_argument('--max_news_len', type=int, default=200,
                            help='News will be padded or truncated to this size.')

    arg_parser.add_argument('--max_hashtag_len', type=int, default=200,
                            help='Hashtag sequences will be padded or truncated to this size.')

    arg_parser.add_argument('--batch_size', type=int, default=100,
                            help='The batch size to use when evaluating on the full dataset.')

    arg_parser.add_argument('--gpu_id', type=str, default="0",
                            help='gpu id to use')

    arg_parser.add_argument('--out_file_path', type=str,
                            help='Output text file path to write the predicted hashtags')

    arg_parser.add_argument('--beam_width', type=int, default=4,
                            help='Number of beams in beam search')

    arg_parser.add_argument('--n_best', type=int, default=1,
                            help='Number of hashtags per tweet to be generated')

    args = arg_parser.parse_args()

    import os

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main(args.test_dir,
         args.model_path,
         args.use_cuda,
         args.max_tweet_len,
         args.max_news_len,
         args.max_hashtag_len,
         args.batch_size,
         args.out_file_path,
         args.beam_width,
         args.n_best
         )

from operator import itemgetter
from models.util import contains_digit, tokens_to_seq
from torch.utils.data import Dataset, DataLoader


class Language(object):
    def __init__(self, tweets, news, vocab_limit=None):
        # Vocabulary is created from tweets, news but not hashtags now.

        self.tweets = tweets
        self.news = news

        self.vocab = self.create_vocab()
        if vocab_limit is None:
            vocab_limit = len(self.vocab)

        truncated_vocab = sorted(self.vocab.items(), key=itemgetter(1), reverse=True)[:vocab_limit]

        self.tok_to_idx = dict()
        self.tok_to_idx['<MSK>'] = 0
        self.tok_to_idx['<SOS>'] = 1
        self.tok_to_idx['<EOS>'] = 2
        self.tok_to_idx['<UNK>'] = 3
        for idx, (tok, _) in enumerate(truncated_vocab):
            self.tok_to_idx[tok] = idx + 4
        self.idx_to_tok = {idx: tok for tok, idx in self.tok_to_idx.items()}

    def create_vocab(self):
        vocab = dict()

        for tweet in self.tweets:
            tokens = tweet.split()
            for token in tokens:
                if not contains_digit(token) and '@' not in token and 'http' not in token and 'www' not in token:
                    vocab[token] = vocab.get(token, 0) + 1

        for news in self.news:
            tokens = news.split()
            for token in tokens:
                if not contains_digit(token) and '@' not in token and 'http' not in token and 'www' not in token:
                    vocab[token] = vocab.get(token, 0) + 1

        return vocab


class TweetNewsDataset(Dataset):
    def __init__(self, data_dir, use_cuda, lang=None, max_tweet_len=128, max_news_len=100, max_hashtag_len=100,
                 vocab_limit=None, use_extended_vocab=True):
        self.data_dir = data_dir
        self.tweet_file = self.data_dir + "/tweets.txt"
        self.news_file = self.data_dir + "/news.txt"
        self.hashtags_file = self.data_dir + "/hashtag.txt"
        self.max_tweet_len = max_tweet_len
        self.max_news_len = max_news_len
        self.max_hashtag_len = max_hashtag_len
        self.use_extended_vocab = use_extended_vocab
        self.use_cuda = use_cuda

        self.tweets = []
        with open(self.tweet_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                mod_line = line.strip()
                self.tweets.append(mod_line)

        self.news = []
        with open(self.news_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                mod_line = line.strip()
                self.news.append(mod_line)

        self.hashtags = []
        with open(self.hashtags_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                mod_line = line.strip()
                self.hashtags.append(mod_line)

        if lang is None:
            lang = Language(self.tweets, self.news, vocab_limit=vocab_limit)

        self.lang = lang
        assert len(self.hashtags) == len(self.news) == len(self.tweets)

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet_token_list = self.tweets[idx].strip().split()
        news_token_list = self.news[idx].strip().split()
        hashtag_token_list = self.hashtags[idx].strip().split()

        tweet_token_list = ['<SOS>'] + tweet_token_list[:self.max_tweet_len - 2] + ['<EOS>']
        news_token_list = ['<SOS>'] + news_token_list[:self.max_news_len - 2] + ['<EOS>']
        hashtag_token_list = ['<SOS>'] + hashtag_token_list[:self.max_hashtag_len - 2] + ['<EOS>']

        tweet_seq = tokens_to_seq(tweet_token_list, self.lang.tok_to_idx, self.max_tweet_len, self.use_extended_vocab)
        news_seq = tokens_to_seq(news_token_list, self.lang.tok_to_idx, self.max_news_len, self.use_extended_vocab)
        output_seq = tokens_to_seq(hashtag_token_list, self.lang.tok_to_idx, self.max_hashtag_len,
                                   self.use_extended_vocab, input_tokens=tweet_token_list + news_token_list)

        if self.use_cuda:
            tweet_seq = tweet_seq.cuda()
            news_seq = news_seq.cuda()
            output_seq = output_seq.cuda()

        return tweet_seq, news_seq, output_seq, ' '.join(tweet_token_list), ' '.join(news_token_list), ' '.join(
            hashtag_token_list)


if __name__ == '__main__':
    # todo make all the tweets in one file
    # todo make all the news in one file
    # todo make all the hashtags in one file with one hashtag per line

    dataset = TweetNewsDataset("./data/train/", use_cuda=False)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    count = 0
    for tweet_seq, news_seq, output_seq, tweet_str, news_str, hashtag_str in data_loader:
        if count == 20:
            break
        print(count)
        print(tweet_seq.size(), tweet_seq, tweet_str)
        print(news_seq.size(), news_seq, news_str)
        print(output_seq.size(), output_seq, hashtag_str)
        count += 1
        print("*" * 80)

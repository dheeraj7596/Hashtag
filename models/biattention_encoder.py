import torch
from torch import nn
import torch.nn.functional as F


class BiAttentionEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size ** 0.5)
        self.gru_tweets = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.gru_news = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)
        self.W_biatt = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.W_tweets = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.W_news = nn.Linear(4 * hidden_size, 2 * hidden_size)
        self.combine_hidden = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, input_tweets, input_news, lengths_tweets, lengths_news):
        # Code follows the naming convention in the paper
        assert input_tweets.data.shape[0] == input_news.data.shape[0]
        batch_size = input_tweets.data.shape[0]
        hidden_tweets = self.init_hidden(batch_size)
        hidden_news = self.init_hidden(batch_size)

        encoded_tweets, hidden_tweets = self.encode(hidden_tweets, input_tweets, lengths_tweets, self.gru_tweets)
        encoded_news, hidden_news = self.encode(hidden_news, input_news, lengths_news, self.gru_news)

        assert encoded_news.shape[2] == encoded_tweets.shape[2]

        transformed_news_hidden = self.W_biatt(encoded_news)
        f_score_tweets_news = torch.bmm(encoded_tweets, transformed_news_hidden.transpose(2, 1))
        alpha_tweets_news = F.softmax(f_score_tweets_news, dim=-1)

        transformed_tweets_hidden = self.W_biatt(encoded_tweets)
        f_score_news_tweets = torch.bmm(encoded_news, transformed_tweets_hidden.transpose(2, 1))
        alpha_news_tweets = F.softmax(f_score_news_tweets, dim=-1)

        r_news = torch.bmm(alpha_tweets_news, encoded_news)
        r_tweets = torch.bmm(alpha_news_tweets, encoded_tweets)

        tweet_outputs = torch.cat([encoded_tweets, r_news], dim=-1)  # [batch_size, tweet_seq_len, 4 * hidden]
        news_outputs = torch.cat([encoded_news, r_tweets], dim=-1)  # [batch_size, news_seq_len, 4 * hidden]

        v_tweets = F.tanh(self.W_tweets(tweet_outputs))  # [batch_size, tweet_seq_len, 2 * hidden]
        v_news = F.tanh(self.W_news(news_outputs))  # [batch_size, news_seq_len, 2 * hidden]

        encoder_output = torch.cat([v_tweets, v_news],
                                   dim=1)  # [batch_size, (tweet_seq_len + news_seq_len), 2 * hidden]
        hidden = self.combine_hidden(torch.cat([hidden_tweets, hidden_news], dim=-1))  # [2, batch_size, hidden]
        return encoder_output, hidden

    def encode(self, hidden, input, lengths, gru):
        # replace OOV words with <UNK> before embedding
        input = input.masked_fill(input > self.embedding.num_embeddings, 3)

        sorted_lengths, lengths_idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, lengths_idx_unsort = torch.sort(lengths_idx_sort, dim=0)

        input = input[lengths_idx_sort, :]
        embedded = self.embedding(input)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
        gru.flatten_parameters()
        output, hidden = gru(packed_embedded, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = output[lengths_idx_unsort, :, :]
        output_lengths = output_lengths[lengths_idx_unsort]
        hidden = hidden[:, lengths_idx_unsort, :]
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.hidden_size)  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden

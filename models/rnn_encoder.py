import torch
from torch import nn


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, self.embedding_size)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size ** 0.5)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input_tweets, input_news, hidden, lengths_tweets, lengths_news):
        # since we are pack_padding, input batch must be sorted by sequence length

        # replace OOV words with <UNK> before embedding
        input_tweets = input_tweets.masked_fill(input_tweets > self.embedding.num_embeddings, 3)

        sorted_tweets_lengths, tweets_lengths_idx_sort = torch.sort(lengths_tweets, dim=0, descending=True)
        _, tweets_lengths_idx_unsort = torch.sort(tweets_lengths_idx_sort, dim=0)

        input_tweets = input_tweets[tweets_lengths_idx_sort, :]

        embedded = self.embedding(input_tweets)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, sorted_tweets_lengths, batch_first=True)
        self.gru.flatten_parameters()
        output, hidden = self.gru(packed_embedded, hidden)
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = output[tweets_lengths_idx_unsort, :, :]
        output_lengths = output_lengths[tweets_lengths_idx_unsort]
        hidden = hidden[:, tweets_lengths_idx_unsort, :]
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.hidden_size)  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden

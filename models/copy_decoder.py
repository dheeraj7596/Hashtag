import torch
from torch import nn
from dataset import Language
import torch.nn.functional as F


class CopyDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, lang: Language, max_tweet_length, max_news_length,
                 max_hashtag_length, tweet_cov_loss_factor=0, news_cov_loss_factor=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_tweet_length = max_tweet_length
        self.max_news_length = max_news_length
        self.max_hashtag_length = max_hashtag_length
        self.tweet_cov_loss_factor = tweet_cov_loss_factor
        self.news_cov_loss_factor = news_cov_loss_factor

        self.embedding = nn.Embedding(len(lang.tok_to_idx), self.embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size ** 0.5)
        self.embedding.weight.data[0, :] = 0.0

        self.gen_attn_W = nn.Linear(self.hidden_size, self.hidden_size)

        self.coverage_tweets = nn.Linear(self.max_tweet_length, self.hidden_size)
        self.coverage_news = nn.Linear(self.max_news_length, self.hidden_size)

        self.gen_context_linear = nn.Linear(self.hidden_size, 1)
        self.state_linear = nn.Linear(self.hidden_size, 1)
        self.input_linear = nn.Linear(self.embedding_size, 1)

        self.out = nn.Linear(self.hidden_size, len(lang.tok_to_idx))

        self.copy_tweets_context_linear = nn.Linear(self.hidden_size, 1)
        self.copy_news_context_linear = nn.Linear(self.hidden_size, 1)

        self.gru = nn.GRU(self.hidden_size + self.embedding_size, self.hidden_size, batch_first=True)

    def forward(self, encoder_outputs, input_tweets, input_news, final_encoder_hidden, targets=None,
                lengths_tweets=None, lengths_news=None, keep_prob=1.0, teacher_forcing=0.0):

        assert lengths_tweets is not None
        assert lengths_news is not None

        input_tweets = input_tweets[:, :lengths_tweets.max()]
        input_news = input_news[:, :lengths_news.max()]

        batch_size = encoder_outputs.data.shape[0]
        tweet_coverage_vec = torch.zeros(batch_size, 1, self.max_tweet_length)
        news_coverage_vec = torch.zeros(batch_size, 1, self.max_news_length)

        hidden = torch.zeros(1, batch_size, self.hidden_size)  # overwrite the encoder hidden state with zeros
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        sos_output = torch.zeros(
            (batch_size, self.embedding.num_embeddings + lengths_tweets.max() + lengths_news.max()))
        sos_output[:, 1] = 1.0  # index 1 is the <SOS> token, one-hot encoding
        sos_idx = torch.ones((batch_size, 1)).long()

        if next(self.parameters()).is_cuda:
            sos_output = sos_output.cuda()
            sos_idx = sos_idx.cuda()

        decoder_outputs = [sos_output]
        sampled_idxs = [sos_idx]

        iput = sos_idx

        dropout_mask = torch.rand(batch_size, 1, self.hidden_size + self.embedding.embedding_dim)
        dropout_mask = dropout_mask <= keep_prob
        dropout_mask = dropout_mask.float() / keep_prob

        coverage_loss = 0

        for step_idx in range(1, self.max_hashtag_length):
            if targets is not None and teacher_forcing > 0.0:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = torch.rand((batch_size, 1)) <= teacher_forcing
                teacher_forcing_mask.requires_grad = False
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                iput = iput.masked_scatter(teacher_forcing_mask, targets[:, step_idx - 1:step_idx])

            output, hidden, copy_tweet_attn_weights, copy_news_attn_weights = self.step(iput, hidden, encoder_outputs,
                                                                                        input_tweets,
                                                                                        input_news,
                                                                                        lengths_tweets,
                                                                                        lengths_news,
                                                                                        tweet_coverage_vec,
                                                                                        news_coverage_vec,
                                                                                        dropout_mask=dropout_mask)

            temp_copy_tweet_attn_weights = torch.zeros_like(tweet_coverage_vec)
            temp_copy_tweet_attn_weights[:, :, :copy_tweet_attn_weights.shape[2]] = copy_tweet_attn_weights
            temp_copy_news_attn_weights = torch.zeros_like(news_coverage_vec)
            temp_copy_news_attn_weights[:, :, :copy_news_attn_weights.shape[2]] = copy_news_attn_weights

            tweet_cov_loss = self.tweet_cov_loss_factor * torch.min(tweet_coverage_vec,
                                                                    temp_copy_tweet_attn_weights).sum()
            news_cov_loss = self.news_cov_loss_factor * torch.min(news_coverage_vec, temp_copy_news_attn_weights).sum()

            coverage_loss = coverage_loss + tweet_cov_loss + news_cov_loss
            tweet_coverage_vec = tweet_coverage_vec + temp_copy_tweet_attn_weights
            news_coverage_vec = news_coverage_vec + temp_copy_news_attn_weights

            decoder_outputs.append(output)
            _, topi = decoder_outputs[-1].topk(1)
            iput = topi.view(batch_size, 1)
            sampled_idxs.append(iput)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)
        coverage_loss = coverage_loss / (self.max_hashtag_length * batch_size)

        # Since we are using NLL loss, returning log probabilities
        return torch.log(decoder_outputs), sampled_idxs, coverage_loss

    def step(self, prev_idx, prev_hidden, encoder_outputs, input_tweets, input_news, lengths_tweets, lengths_news,
             tweet_coverage_vec, news_coverage_vec, dropout_mask=None):
        batch_size = prev_idx.shape[0]
        vocab_size = len(self.lang.tok_to_idx)

        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3  # 3 is unk_token index
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask,
                                           unks)  # replace copied tokens with <UNK> token before embedding

        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        # encoder_output * W * decoder_hidden for each encoder_output
        transformed_all_prev_hidden = self.gen_attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        coverage_tweets_attn = self.coverage_tweets(tweet_coverage_vec).view(batch_size, self.hidden_size, 1)
        coverage_news_attn = self.coverage_news(news_coverage_vec).view(batch_size, self.hidden_size, 1)

        gen_scores = torch.bmm(encoder_outputs,
                               transformed_all_prev_hidden + coverage_tweets_attn + coverage_news_attn).squeeze(2)
        attn_weights = F.softmax(gen_scores, dim=1).unsqueeze(
            1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(attn_weights, encoder_outputs)  # weighted sum of encoder_outputs (i.e. values)

        rnn_input = torch.cat((context, embedded), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        output, hidden = self.gru(rnn_input, prev_hidden)
        output = self.out(output.squeeze(1))
        output = F.softmax(output, dim=1)
        append_for_copy = torch.zeros((batch_size, lengths_tweets.max() + lengths_news.max()))
        output = torch.cat([output, append_for_copy], dim=-1)

        p_gen = F.sigmoid(self.gen_context_linear(context) +
                          self.state_linear(prev_hidden).view(batch_size, 1, -1) +
                          self.input_linear(embedded)).squeeze(1)

        assert encoder_outputs.shape[1] == (lengths_tweets.max() + lengths_news.max())
        tweet_encoder_outputs = encoder_outputs[:, :lengths_tweets.max(), :]
        news_encoder_outputs = encoder_outputs[:, lengths_tweets.max():, :]

        copy_tweet_attn_weights = attn_weights[:, :, :lengths_tweets.max()]
        copy_news_attn_weights = attn_weights[:, :, lengths_tweets.max():]

        p_copy_tweets = self.copy_from_source(tweet_encoder_outputs,
                                              copy_tweet_attn_weights,
                                              self.copy_tweets_context_linear).squeeze(1)
        p_copy_news = self.copy_from_source(news_encoder_outputs,
                                            copy_news_attn_weights,
                                            self.copy_news_context_linear).squeeze(1)

        temp = torch.cat([p_gen, p_copy_tweets, p_copy_news], dim=-1)
        temp = F.softmax(temp, dim=-1)
        p_gen = temp[:, 0].unsqueeze(1)
        p_copy_tweets = temp[:, 1].unsqueeze(1)
        p_copy_news = temp[:, 2].unsqueeze(1)

        probs = torch.zeros(batch_size, vocab_size + lengths_tweets.max() + lengths_news.max())
        tweet_copy_probabilities = torch.zeros_like(probs)
        tweet_copy_probabilities.scatter_add_(1, input_tweets, copy_tweet_attn_weights.squeeze(1))

        news_copy_probabilities = torch.zeros_like(probs)
        news_copy_probabilities.scatter_add_(1, input_news, copy_news_attn_weights.squeeze(1))

        probs = p_gen * output + p_copy_tweets * tweet_copy_probabilities + p_copy_news * news_copy_probabilities

        return probs, hidden, copy_tweet_attn_weights, copy_news_attn_weights

    def copy_from_source(self, src_encoder_outputs, src_attn_weights, copy_from_src_linear):
        context = torch.bmm(src_attn_weights, src_encoder_outputs)  # weighted sum of encoder_outputs (i.e. values)
        p_copy = F.sigmoid(copy_from_src_linear(context))
        return p_copy

    def init_hidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.hidden_size)
        if next(self.parameters()).is_cuda:
            return result.cuda()
        else:
            return result
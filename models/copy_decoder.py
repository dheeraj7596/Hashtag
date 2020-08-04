import torch
from torch import nn
from dataset import Language
import torch.nn.functional as F


class CopyDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, lang: Language, max_tweet_length, max_news_length,
                 max_hashtag_length, decode_strategy, tweet_cov_loss_factor=0, news_cov_loss_factor=0):
        super().__init__()
        self.EPS = 1e-8
        self.EOS_ID = 2
        self.min_length = 3  # this includes <SOS>, <EOS>
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_tweet_length = max_tweet_length
        self.max_news_length = max_news_length
        self.max_hashtag_length = max_hashtag_length
        self.tweet_cov_loss_factor = tweet_cov_loss_factor
        self.news_cov_loss_factor = news_cov_loss_factor
        self.decode_strategy = decode_strategy

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

    def forward(self, encoder_outputs, input_tweets, input_news, final_encoder_hidden, beam_width, targets=None,
                lengths_tweets=None, lengths_news=None, keep_prob=1.0, teacher_forcing=0.0):
        if self.decode_strategy == "beam":
            return self.beam_decode(encoder_outputs,
                                    input_tweets,
                                    input_news,
                                    final_encoder_hidden,
                                    beam_width,
                                    lengths_tweets=lengths_tweets,
                                    lengths_news=lengths_news,
                                    targets=targets)

        elif self.decode_strategy == "greedy":
            return self.greedy_decode(encoder_outputs,
                                      input_tweets,
                                      input_news,
                                      final_encoder_hidden,
                                      lengths_tweets=lengths_tweets,
                                      lengths_news=lengths_news,
                                      targets=targets,
                                      teacher_forcing=teacher_forcing)
        else:
            raise ValueError("decoder_mode must be 'beam' or 'greedy'")

    def beam_decode(self, encoder_outputs, input_tweets, input_news, final_encoder_hidden, beam_width, targets=None,
                    lengths_tweets=None, lengths_news=None, keep_prob=1.0):

        def sort_hyps(list_of_hyps):
            return sorted(list_of_hyps, key=lambda x: sum(x["logprobs"]) / len(x["logprobs"]), reverse=True)

        assert lengths_tweets is not None
        assert lengths_news is not None

        input_tweets = input_tweets[:, :lengths_tweets.max()]
        input_news = input_news[:, :lengths_news.max()]

        batch_size = encoder_outputs.data.shape[0]
        final_decoder_outputs = torch.zeros(batch_size, self.max_hashtag_length,
                                            self.embedding.num_embeddings + lengths_tweets.max() + lengths_news.max())
        final_sampled_idxs = torch.zeros(batch_size, self.max_hashtag_length, 1)
        final_coverage_loss = 0

        if next(self.parameters()).is_cuda:
            final_decoder_outputs = final_decoder_outputs.cuda()
            final_sampled_idxs = final_sampled_idxs.cuda()

        for b_index in range(batch_size):
            sent_encoder_outputs = encoder_outputs[b_index, :, :].unsqueeze(0)
            sent_input_tweet = input_tweets[b_index, :].unsqueeze(0)
            sent_input_news = input_news[b_index, :].unsqueeze(0)
            hidden = torch.zeros(1, 1, self.hidden_size)  # overwrite the encoder hidden state with zeros
            sent_tweet_coverage_vec = torch.zeros(1, 1, self.max_tweet_length)
            sent_news_coverage_vec = torch.zeros(1, 1, self.max_news_length)
            sent_coverage_loss = 0
            if next(self.parameters()).is_cuda:
                hidden = hidden.cuda()
                sent_tweet_coverage_vec = sent_tweet_coverage_vec.cuda()
                sent_news_coverage_vec = sent_news_coverage_vec.cuda()

            # every decoder output seq starts with <SOS>
            sos_output = torch.zeros((1, self.embedding.num_embeddings + lengths_tweets.max() + lengths_news.max()))
            sos_output[:, 1] = 1  # index 1 is the <SOS> token, one-hot encoding
            sos_idx = torch.ones((1, 1)).long()

            dropout_mask = torch.rand(1, 1, self.hidden_size + self.embedding.embedding_dim)
            dropout_mask = dropout_mask <= keep_prob
            dropout_mask = dropout_mask.float() / keep_prob

            if next(self.parameters()).is_cuda:
                sos_output = sos_output.cuda()
                sos_idx = sos_idx.cuda()

            hypothesis = [
                {
                    "dec_state": hidden,
                    "sampled_idxs": [sos_idx],
                    "logprobs": [0],
                    "decoder_outputs": [sos_output],
                    "coverage_loss": sent_coverage_loss,
                    "tweet_coverage_vec": sent_tweet_coverage_vec,
                    "news_coverage_vec": sent_news_coverage_vec
                }
            ]

            finished_hypothesis = []
            for step_idx in range(1, self.max_hashtag_length):
                if len(finished_hypothesis) >= beam_width:
                    break
                new_hypothesis = []
                for hyp in hypothesis:
                    out_idxs = hyp["sampled_idxs"]
                    iput = hyp["sampled_idxs"][-1]
                    prev_hidden = hyp["dec_state"]
                    old_logprobs = hyp["logprobs"]
                    old_decoder_outputs = hyp["decoder_outputs"]
                    old_coverage_loss = hyp["coverage_loss"]
                    sent_tweet_coverage_vec = hyp["tweet_coverage_vec"]
                    sent_news_coverage_vec = hyp["news_coverage_vec"]

                    output, hidden, copy_tweet_attn_weights, copy_news_attn_weights = self.step(iput, prev_hidden,
                                                                                                sent_encoder_outputs,
                                                                                                sent_input_tweet,
                                                                                                sent_input_news,
                                                                                                lengths_tweets,
                                                                                                lengths_news,
                                                                                                sent_tweet_coverage_vec,
                                                                                                sent_news_coverage_vec,
                                                                                                dropout_mask=dropout_mask)

                    temp_copy_tweet_attn_weights = torch.zeros_like(sent_tweet_coverage_vec)
                    temp_copy_tweet_attn_weights[:, :, :copy_tweet_attn_weights.shape[2]] = copy_tweet_attn_weights
                    temp_copy_news_attn_weights = torch.zeros_like(sent_news_coverage_vec)
                    temp_copy_news_attn_weights[:, :, :copy_news_attn_weights.shape[2]] = copy_news_attn_weights

                    tweet_cov_loss = self.tweet_cov_loss_factor * torch.min(sent_tweet_coverage_vec,
                                                                            temp_copy_tweet_attn_weights).sum()
                    news_cov_loss = self.news_cov_loss_factor * torch.min(sent_news_coverage_vec,
                                                                          temp_copy_news_attn_weights).sum()

                    new_coverage_loss = old_coverage_loss + tweet_cov_loss + news_cov_loss
                    new_sent_tweet_coverage_vec = sent_tweet_coverage_vec + temp_copy_tweet_attn_weights
                    new_news_coverage_vec = sent_news_coverage_vec + temp_copy_news_attn_weights
                    probs, indices = torch.topk(output, dim=-1, k=beam_width)
                    for i in range(beam_width):
                        p = probs[:, i]
                        idx = indices[:, i]
                        new_dict = {
                            "dec_state": hidden,
                            "sampled_idxs": out_idxs + [idx.unsqueeze(1)],
                            "logprobs": old_logprobs + [float(torch.log(p).detach().cpu().numpy())],
                            "decoder_outputs": old_decoder_outputs + [output],
                            "coverage_loss": new_coverage_loss,
                            "tweet_coverage_vec": new_sent_tweet_coverage_vec,
                            "news_coverage_vec": new_news_coverage_vec
                        }
                        new_hypothesis.append(new_dict)

                # time to pick the best of new hypotheses
                sorted_new_hypothesis = sort_hyps(new_hypothesis)
                hypothesis = []
                for hyp in sorted_new_hypothesis:
                    if hyp["sampled_idxs"][-1] == self.EOS_ID:
                        if len(hyp["sampled_idxs"]) >= self.min_length:
                            finished_hypothesis.append(hyp)
                    else:
                        hypothesis.append(hyp)
                    if len(hypothesis) == beam_width or len(finished_hypothesis) == beam_width:
                        break

            if len(finished_hypothesis) > 0:
                final_candidates = finished_hypothesis
            else:
                final_candidates = hypothesis

            sorted_final_candidates = sort_hyps(final_candidates)
            best_candidate = sorted_final_candidates[0]
            sent_decoder_outputs = torch.stack(best_candidate["decoder_outputs"], dim=0).squeeze(1)
            sent_sampled_idxs = torch.stack(best_candidate["sampled_idxs"], dim=0).squeeze(1)

            final_decoder_outputs[b_index, :sent_decoder_outputs.shape[0], :] = sent_decoder_outputs
            final_sampled_idxs[b_index, :sent_sampled_idxs.shape[0], :] = sent_sampled_idxs
            final_coverage_loss += best_candidate["coverage_loss"]

        final_coverage_loss = final_coverage_loss / (self.max_hashtag_length * batch_size)
        # Since we are using NLL loss, returning log probabilities
        return torch.log(final_decoder_outputs + self.EPS), final_sampled_idxs, final_coverage_loss

    def greedy_decode(self, encoder_outputs, input_tweets, input_news, final_encoder_hidden, targets=None,
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
            tweet_coverage_vec = tweet_coverage_vec.cuda()
            news_coverage_vec = news_coverage_vec.cuda()
        else:
            hidden = hidden
            tweet_coverage_vec = tweet_coverage_vec
            news_coverage_vec = news_coverage_vec

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
        return torch.log(decoder_outputs + self.EPS), sampled_idxs, coverage_loss

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

        rnn_input = torch.cat((embedded, context), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        output, hidden = self.gru(rnn_input, prev_hidden)
        output = self.out(output.squeeze(1))
        output = F.softmax(output, dim=1)
        append_for_copy = torch.zeros((batch_size, lengths_tweets.max() + lengths_news.max()))
        if next(self.parameters()).is_cuda:
            append_for_copy = append_for_copy.cuda()
        output = torch.cat([output, append_for_copy], dim=-1)

        p_gen = (self.gen_context_linear(context) +
                 self.state_linear(prev_hidden).view(batch_size, 1, -1) +
                 self.input_linear(embedded)).squeeze(1)

        assert encoder_outputs.shape[1] == (lengths_tweets.max() + lengths_news.max())
        tweet_encoder_outputs = encoder_outputs[:, :lengths_tweets.max(), :]
        news_encoder_outputs = encoder_outputs[:, lengths_tweets.max():, :]

        copy_tweet_attn_weights = F.softmax(attn_weights[:, :, :lengths_tweets.max()], dim=-1)
        copy_news_attn_weights = F.softmax(attn_weights[:, :, lengths_tweets.max():], dim=-1)

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
        if next(self.parameters()).is_cuda:
            probs = probs.cuda()
        tweet_copy_probabilities = torch.zeros_like(probs)
        tweet_copy_probabilities.scatter_add_(1, input_tweets, copy_tweet_attn_weights.squeeze(1))

        news_copy_probabilities = torch.zeros_like(probs)
        news_copy_probabilities.scatter_add_(1, input_news, copy_news_attn_weights.squeeze(1))

        probs = p_gen * output + p_copy_tweets * tweet_copy_probabilities + p_copy_news * news_copy_probabilities

        return probs, hidden, copy_tweet_attn_weights, copy_news_attn_weights

    def copy_from_source(self, src_encoder_outputs, src_attn_weights, copy_from_src_linear):
        context = torch.bmm(src_attn_weights, src_encoder_outputs)  # weighted sum of encoder_outputs (i.e. values)
        p_copy = copy_from_src_linear(context)
        return p_copy

    def init_hidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.hidden_size)
        if next(self.parameters()).is_cuda:
            return result.cuda()
        else:
            return result

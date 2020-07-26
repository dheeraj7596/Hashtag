import torch
import math
from torch import nn
import torch.nn.functional as F
from dataset import Language


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, lang: Language, max_hashtag_length, decode_strategy, beam_width):
        super(AttentionDecoder, self).__init__()
        self.EPS = 1e-8
        self.EOS_ID = 2
        self.min_length = 3  # this includes <SOS>, <EOS>
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lang = lang
        self.max_hashtag_length = max_hashtag_length
        self.decode_strategy = decode_strategy
        self.beam_width = beam_width

        self.embedding = nn.Embedding(len(lang.tok_to_idx), self.embedding_size, padding_idx=0)
        self.embedding.weight.data.normal_(0, 1 / self.embedding_size ** 0.5)
        self.embedding.weight.data[0, :] = 0.0

        self.attn_W = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size + self.embedding_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, len(lang.tok_to_idx))

    def forward(self, encoder_outputs, input_tweets, input_news, final_encoder_hidden, targets=None,
                lengths_tweets=None, lengths_news=None, keep_prob=1.0, teacher_forcing=0.0, beam_width=4):

        if self.decode_strategy == "beam":
            return self.beam_decode(encoder_outputs,
                                    input_tweets,
                                    input_news,
                                    final_encoder_hidden,
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

    def beam_decode(self, encoder_outputs, input_tweets, input_news, final_encoder_hidden, targets=None,
                    lengths_tweets=None, lengths_news=None, keep_prob=1.0):

        def sort_hyps(list_of_hyps):
            return sorted(list_of_hyps, key=lambda x: sum(x["logprobs"]) / len(x["logprobs"]), reverse=True)

        batch_size = encoder_outputs.data.shape[0]
        final_decoder_outputs = torch.zeros(batch_size, self.max_hashtag_length, self.embedding.num_embeddings)
        # since final_decoder_outputs contain final log softmax probs, initializing it with log of epsilon.
        final_decoder_outputs.fill_(math.log(self.EPS))
        final_sampled_idxs = torch.zeros(batch_size, self.max_hashtag_length, 1)

        if next(self.parameters()).is_cuda:
            final_decoder_outputs = final_decoder_outputs.cuda()
            final_sampled_idxs = final_sampled_idxs.cuda()

        for b_index in range(batch_size):
            sent_encoder_outputs = encoder_outputs[b_index, :, :].unsqueeze(0)
            hidden = torch.zeros(1, 1, self.hidden_size)  # overwrite the encoder hidden state with zeros
            if next(self.parameters()).is_cuda:
                hidden = hidden.cuda()
            else:
                hidden = hidden

            # every decoder output seq starts with <SOS>
            sos_output = torch.zeros((1, self.embedding.num_embeddings))
            sos_output.fill_(math.log(self.EPS))
            sos_output[:, 1] = 0  # index 1 is the <SOS> token, one-hot encoding
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
                    "decoder_outputs": [sos_output]
                }
            ]

            finished_hypothesis = []
            for step_idx in range(1, self.max_hashtag_length):
                if len(finished_hypothesis) >= self.beam_width:
                    break
                new_hypothesis = []
                for hyp in hypothesis:
                    out_idxs = hyp["sampled_idxs"]
                    iput = hyp["sampled_idxs"][-1]
                    prev_hidden = hyp["dec_state"]
                    old_logprobs = hyp["logprobs"]
                    old_decoder_outputs = hyp["decoder_outputs"]

                    output, hidden = self.step(iput, prev_hidden, sent_encoder_outputs, dropout_mask=dropout_mask)
                    probs, indices = torch.topk(output, dim=-1, k=self.beam_width)
                    for i in range(self.beam_width):
                        p = probs[:, i]
                        idx = indices[:, i]
                        new_dict = {
                            "dec_state": hidden,
                            "sampled_idxs": out_idxs + [idx.unsqueeze(1)],
                            "logprobs": old_logprobs + [float(p.detach().cpu().numpy())],
                            "decoder_outputs": old_decoder_outputs + [output]
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
                    if len(hypothesis) == self.beam_width or len(finished_hypothesis) == self.beam_width:
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

        return final_decoder_outputs, final_sampled_idxs, 0

    def greedy_decode(self, encoder_outputs, input_tweets, input_news, final_encoder_hidden, targets=None,
                      lengths_tweets=None, lengths_news=None, keep_prob=1.0, teacher_forcing=0.0):
        batch_size = encoder_outputs.data.shape[0]

        hidden = torch.zeros(1, batch_size, self.hidden_size)  # overwrite the encoder hidden state with zeros
        if next(self.parameters()).is_cuda:
            hidden = hidden.cuda()
        else:
            hidden = hidden

        # every decoder output seq starts with <SOS>
        sos_output = torch.zeros((batch_size, self.embedding.num_embeddings))
        sos_output.fill_(math.log(self.EPS))
        sos_output[:, 1] = 0  # index 1 is the <SOS> token, one-hot encoding
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

        for step_idx in range(1, self.max_hashtag_length):
            if targets is not None and teacher_forcing > 0.0:
                # replace some inputs with the targets (i.e. teacher forcing)
                teacher_forcing_mask = torch.rand((batch_size, 1)) <= teacher_forcing
                teacher_forcing_mask.requires_grad = False
                if next(self.parameters()).is_cuda:
                    teacher_forcing_mask = teacher_forcing_mask.cuda()
                iput = iput.masked_scatter(teacher_forcing_mask, targets[:, step_idx - 1:step_idx])

            output, hidden = self.step(iput, hidden, encoder_outputs, dropout_mask=dropout_mask)

            decoder_outputs.append(output)
            _, topi = decoder_outputs[-1].topk(1)
            iput = topi.view(batch_size, 1)
            sampled_idxs.append(iput)

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        sampled_idxs = torch.stack(sampled_idxs, dim=1)

        return decoder_outputs, sampled_idxs, 0

    def step(self, prev_idx, prev_hidden, encoder_outputs, dropout_mask=None):

        batch_size = prev_idx.shape[0]
        vocab_size = len(self.lang.tok_to_idx)

        out_of_vocab_mask = prev_idx > vocab_size  # [b, 1] bools indicating which seqs copied on the previous step
        unks = torch.ones_like(prev_idx).long() * 3  # 3 is unk_token index
        prev_idx = prev_idx.masked_scatter(out_of_vocab_mask,
                                           unks)  # replace copied tokens with <UNK> token before embedding

        embedded = self.embedding(prev_idx)  # embed input (i.e. previous output token)

        # encoder_output * W * decoder_hidden for each encoder_output
        transformed_hidden = self.attn_W(prev_hidden).view(batch_size, self.hidden_size, 1)
        # reduce encoder outputs and hidden to get scores. remove singleton dimension from multiplication.
        scores = torch.bmm(encoder_outputs, transformed_hidden).squeeze(2)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(1)  # apply softmax to scores to get normalized weights
        context = torch.bmm(attn_weights, encoder_outputs)  # weighted sum of encoder_outputs (i.e. values)

        rnn_input = torch.cat((embedded, context), dim=2)
        if dropout_mask is not None:
            if next(self.parameters()).is_cuda:
                dropout_mask = dropout_mask.cuda()
            rnn_input *= dropout_mask

        output, hidden = self.gru(rnn_input, prev_hidden)

        output = self.out(output.squeeze(1))  # linear transformation to output size
        output = F.log_softmax(output, dim=1)  # log softmax non-linearity to convert to log probabilities

        return output, hidden

    def init_hidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.hidden_size)
        if next(self.parameters()).is_cuda:
            return result.cuda()
        else:
            return result

import argparse
import os
import time
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AdamW
from dataset import TweetNewsDataset
from models.encoder_decoder import EncoderDecoder
from evaluate import evaluate
from models.util import to_np, trim_seqs

from tensorboardX import SummaryWriter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def train(encoder_decoder: EncoderDecoder,
          model_dump_path,
          train_data_loader: DataLoader,
          model_name,
          val_data_loader: DataLoader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          max_length,
          early_stopping,
          patience,
          beam_width):
    global_step = 0
    loss_function = torch.nn.NLLLoss(ignore_index=0)
    optimizer = AdamW(encoder_decoder.parameters(),
                      lr=lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # optimizer = optim.Adam(encoder_decoder.parameters(), lr=lr)
    model_path = model_dump_path + model_name + '/'
    history = {
        'val_loss': [],
        'best_epoch': -1,
        'best_loss': float("inf"),
        'prev_loss': float("inf")
    }

    for epoch, teacher_forcing in enumerate(teacher_forcing_schedule):
        print('epoch %i' % epoch, flush=True)

        for batch_idx, (tweet_idxs, news_idxs, target_idxs, tweet_tokens, news_tokens, hashtag_tokens) in enumerate(
                tqdm(train_data_loader)):
            # tweet_idxs have dim (batch_size x max_tweet_len)
            # news_idxs have dim (batch_size x max_news_len)
            # hashtag_idxs have dim (batch_size x max_hashtag_len)

            lengths_tweets = (tweet_idxs != 0).long().sum(dim=1)
            lengths_news = (news_idxs != 0).long().sum(dim=1)

            optimizer.zero_grad()
            output_log_probs, output_seqs, cov_loss = encoder_decoder(tweet_idxs,
                                                                      news_idxs,
                                                                      lengths_tweets,
                                                                      lengths_news,
                                                                      beam_width,
                                                                      targets=target_idxs,
                                                                      keep_prob=keep_prob,
                                                                      teacher_forcing=teacher_forcing)

            batch_size = tweet_idxs.shape[0]

            flattened_outputs = output_log_probs.contiguous().view(batch_size * max_length, -1)

            batch_loss = loss_function(flattened_outputs, target_idxs.contiguous().view(-1)) + cov_loss
            batch_loss.backward()
            optimizer.step()

            batch_outputs = trim_seqs(output_seqs)

            batch_targets = [[list(seq[seq > 0])] for seq in list(to_np(target_idxs))]

            batch_bleu_score = corpus_bleu(batch_targets, batch_outputs, smoothing_function=SmoothingFunction().method1)

            if global_step < 10 or (global_step % 10 == 0 and global_step < 100) or (
                    global_step % 100 == 0):
                tweet_string = "do you think brett kavanaugh should be confirmed as a justice on the supreme court"
                news_string = "leading catholic publication turns on brett kavanaugh says his nomination to the supreme court should be withdrawn"
                output_string = encoder_decoder.get_response(tweet_string, news_string)
                writer.add_text('kavanaugh', output_string, global_step=global_step)
                print("Global Step: ", global_step, ' kavanaugh ', output_string)

            if global_step % 100 == 0:
                writer.add_scalar('train_batch_loss', batch_loss, global_step)
                writer.add_scalar('train_batch_bleu_score', batch_bleu_score, global_step)
                print("Global Step: ", global_step, ' train_batch_loss ', batch_loss)
                print("Global Step: ", global_step, ' train_batch_bleu_score ', batch_bleu_score)

                # for tag, value in encoder_decoder.named_parameters():
                #     tag = tag.replace('.', '/')
                #     writer.add_histogram('weights/' + tag, value, global_step, bins='doane')
                #     writer.add_histogram('grads/' + tag, to_np(value.grad), global_step, bins='doane')

            global_step += 1

        with torch.no_grad():
            val_loss, val_bleu_score = evaluate(encoder_decoder, val_data_loader)
            history["val_loss"].append(val_loss)

        writer.add_scalar('val_loss', val_loss, global_step=global_step)
        writer.add_scalar('val_bleu_score', val_bleu_score, global_step=global_step)

        encoder_embeddings = encoder_decoder.encoder.embedding.weight.data
        encoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(encoder_embeddings, metadata=encoder_vocab, global_step=0, tag='encoder_embeddings')

        decoder_embeddings = encoder_decoder.decoder.embedding.weight.data
        decoder_vocab = encoder_decoder.lang.tok_to_idx.keys()
        writer.add_embedding(decoder_embeddings, metadata=decoder_vocab, global_step=0, tag='decoder_embeddings')

        tweet_string = "should ask dr ford 1 important question why did you go upstairs to go to the bathroom 99 of 2 story houses have a bathroom downstairs common sense bedrooms are upstairs cbc news fox news real donald trump cnn"
        news_string = "america supreme court brett kavanuagh senate judiciary committee christine blasey ford cnn fox news supreme court bill clinton kavanaugh new york ford donald trump rod rosenstein new york times trump trump maine alaska tara d sonenshine us george washington university elliott school of international affairs christine blasey ford senate judiciary committee fox news ford fox news chris wallace ford wallace bret baier ford brit hume fox news bret baier ford chris wallace david mack andrew napolitano ford rachel mitchell fox news andrew napolitano ford rachel mitchell keith boykin fox kavanaugh ford ford twitter america supreme court brett kavanuagh senate judiciary committee christine blasey ford cnn fox news supreme court bill clinton kavanaugh new york ford donald trump rod rosenstein new york times trump trump maine alaska tara d sonenshine us george washington university elliott school of international affairs fox news sean hannity donald trump fox news fox news cnn msnbc pbs news gallupknight foundation gallupknight foundation gallup knight foundation christine blasey ford senate judiciary committee fox news ford fox news chris wallace ford wallace bret baier ford brit hume fox news bret baier ford chris wallace"
        output_string = encoder_decoder.get_response(tweet_string, news_string)
        writer.add_text('christine blasey ford', output_string, global_step=global_step)
        print("Global Step: ", global_step, ' christine blasey ford ', output_string)

        print('val loss: %.5f, val BLEU score: %.5f' % (val_loss, val_bleu_score), flush=True)
        torch.save(encoder_decoder, "%s%s_%i.pt" % (model_path, model_name, epoch))

        print('-' * 100, flush=True)

        if history['val_loss'][-1] < history['best_loss'] or history['val_loss'][-1] < history['prev_loss']:
            history['best_loss'] = history['val_loss'][-1]
            history['best_epoch'] = epoch
        elif early_stopping and epoch - history['best_epoch'] > patience:
            # early stopping
            print("Early stopping at epoch {0}, best result at epoch {1}".format(epoch, history['best_epoch']))
            break
        history['prev_loss'] = history['val_loss'][-1]


def main(model_name, model_dump_path, train_dir, val_dir, use_cuda, batch_size, teacher_forcing_schedule, keep_prob,
         lr, encoder_type, decoder_type, decode_strategy, beam_width, max_tweet_len, max_news_len, max_hashtag_len,
         vocab_limit, hidden_size, embedding_size, tweet_cov_loss_factor, news_cov_loss_factor, early_stopping,
         patience, seed=42):
    model_path = model_dump_path + model_name + '/'

    print("training %s with use_cuda=%s, batch_size=%i" % (model_name, use_cuda, batch_size), flush=True)
    print("teacher_forcing_schedule=", teacher_forcing_schedule, flush=True)
    print(
        "train_dir=%s, val_dir=%s, keep_prob=%f, lr=%f, encoder_type=%s, decoder_type=%s, decode_strategy=%s, beam_width=%i, vocab_limit=%i, hidden_size=%i, embedding_size=%i, max_tweetlength=%i, max_newslength=%i, max_hashtaglength=%i, tweet_cov_loss_factor=%f, news_cov_loss_factor=%f, patience=%i, seed=%i" % (
            train_dir, val_dir, keep_prob, lr, encoder_type, decoder_type, decode_strategy, beam_width, vocab_limit,
            hidden_size, embedding_size, max_tweet_len, max_news_len, max_hashtag_len, tweet_cov_loss_factor,
            news_cov_loss_factor, patience, seed),
        flush=True)

    if os.path.isdir(model_path):

        print("loading encoder and decoder from model_path", flush=True)
        encoder_decoder = torch.load(model_path + model_name + '.pt')

        print("creating training and validation datasets with saved languages", flush=True)
        train_dataset = TweetNewsDataset(data_dir=train_dir,
                                         use_cuda=use_cuda,
                                         lang=encoder_decoder.lang,
                                         max_tweet_len=max_tweet_len,
                                         max_news_len=max_news_len,
                                         max_hashtag_len=max_hashtag_len,
                                         vocab_limit=vocab_limit,
                                         use_extended_vocab=(encoder_decoder.decoder_type == 'copy'))

        val_dataset = TweetNewsDataset(data_dir=val_dir,
                                       use_cuda=use_cuda,
                                       lang=encoder_decoder.lang,
                                       max_tweet_len=max_tweet_len,
                                       max_news_len=max_news_len,
                                       max_hashtag_len=max_hashtag_len,
                                       vocab_limit=vocab_limit,
                                       use_extended_vocab=(encoder_decoder.decoder_type == 'copy'))

    else:
        os.mkdir(model_path)

        print("creating training and validation datasets", flush=True)

        train_dataset = TweetNewsDataset(data_dir=train_dir,
                                         use_cuda=use_cuda,
                                         max_tweet_len=max_tweet_len,
                                         max_news_len=max_news_len,
                                         max_hashtag_len=max_hashtag_len,
                                         vocab_limit=vocab_limit,
                                         use_extended_vocab=(decoder_type == 'copy'))

        val_dataset = TweetNewsDataset(data_dir=val_dir,
                                       use_cuda=use_cuda,
                                       lang=train_dataset.lang,
                                       max_tweet_len=max_tweet_len,
                                       max_news_len=max_news_len,
                                       max_hashtag_len=max_hashtag_len,
                                       vocab_limit=vocab_limit,
                                       use_extended_vocab=(decoder_type == 'copy'))

        print("creating encoder-decoder model", flush=True)
        encoder_decoder = EncoderDecoder(lang=train_dataset.lang,
                                         max_tweet_length=max_tweet_len,
                                         max_news_length=max_news_len,
                                         max_hashtag_length=max_hashtag_len,
                                         hidden_size=hidden_size,
                                         embedding_size=embedding_size,
                                         encoder_type=encoder_type,
                                         decoder_type=decoder_type,
                                         decode_strategy=decode_strategy,
                                         tweet_cov_loss_factor=tweet_cov_loss_factor,
                                         news_cov_loss_factor=news_cov_loss_factor
                                         )

        torch.save(encoder_decoder, model_path + '/%s.pt' % model_name)

    if use_cuda:
        encoder_decoder = encoder_decoder.cuda()
    else:
        encoder_decoder = encoder_decoder.cpu()

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    train(encoder_decoder,
          model_dump_path,
          train_data_loader,
          model_name,
          val_data_loader,
          keep_prob,
          teacher_forcing_schedule,
          lr,
          encoder_decoder.decoder.max_hashtag_length,
          early_stopping,
          patience,
          beam_width)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse training parameters')
    parser.add_argument('--model_name', type=str,
                        help='the name of a subdirectory of ./model/ that '
                             'contains encoder and decoder model files')

    parser.add_argument('--epochs', type=int, default=50,
                        help='the number of epochs to train')

    parser.add_argument('--use_cuda', action='store_true',
                        help='flag indicating that cuda will be used')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples in a batch')

    parser.add_argument('--teacher_forcing_fraction', type=float, default=0.5,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--train_dir', type=str, default="./data/train",
                        help='training directory which contains tweets.txt, news.txt, hashtag.txt')

    parser.add_argument('--val_dir', type=str, default="./data/val",
                        help='validation directory which contains tweets.txt, news.txt, hashtag.txt')

    parser.add_argument('--model_dump_path', type=str,
                        help='The name of a directory of contains encoder and decoder model files.')

    parser.add_argument('--log_dir', type=str,
                        help='The path to logs directory.')

    parser.add_argument('--scheduled_teacher_forcing', action='store_true',
                        help='Linearly decrease the teacher forcing fraction '
                             'from 1.0 to 0.0 over the specified number of epocs')

    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable/Disable earlystopping')

    parser.add_argument('--patience', type=int, default=3,
                        help="Patience in early stopping")

    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='Probablity of keeping an element in the dropout step.')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')

    parser.add_argument('--encoder_type', type=str, default='biattention',
                        help="Allowed values 'rnn' or 'biattention'")

    parser.add_argument('--decoder_type', type=str, default='copy',
                        help="Allowed values 'copy' or 'attn'")

    parser.add_argument('--decode_strategy', type=str, default='greedy',
                        help="Allowed values 'beam' or 'greedy'")

    parser.add_argument('--beam_width', type=int, default=4,
                        help="Beam size.")

    parser.add_argument('--max_tweet_len', type=int, default=200,
                        help="Tweets will be padded or truncated to this size.")

    parser.add_argument('--max_news_len', type=int, default=200,
                        help='News will be padded or truncated to this size.')

    parser.add_argument('--max_hashtag_len', type=int, default=10,
                        help='Hashtag sequences will be padded or truncated to this size.')

    parser.add_argument('--vocab_limit', type=int, default=10000,
                        help='When creating a new Language object the vocab'
                             'will be truncated to the most frequently'
                             'occurring words in the training dataset.')

    parser.add_argument('--hidden_size', type=int, default=256,
                        help='The number of RNN units in the encoder. 2x this '
                             'number of RNN units will be used in the decoder')

    parser.add_argument('--embedding_size', type=int, default=128,
                        help='Embedding size used in both encoder and decoder')

    parser.add_argument('--tweet_cov_loss_factor', type=float, default=1,
                        help='Tweet coverage loss factor')

    parser.add_argument('--news_cov_loss_factor', type=float, default=1,
                        help='News coverage loss factor')

    parser.add_argument('--gpu_id', type=str, default="0",
                        help='gpu id to use')

    args = parser.parse_args()

    import os

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    writer = SummaryWriter(args.log_dir + '%s_%s' % (args.model_name, str(int(time.time()))))
    if args.scheduled_teacher_forcing:
        schedule = np.arange(1.0, 0.0, -1.0 / args.epochs)
    else:
        schedule = np.ones(args.epochs) * args.teacher_forcing_fraction

    main(args.model_name, args.model_dump_path, args.train_dir, args.val_dir, args.use_cuda, args.batch_size,
         schedule, args.keep_prob, args.lr, args.encoder_type, args.decoder_type, args.decode_strategy,
         args.beam_width, args.max_tweet_len, args.max_news_len, args.max_hashtag_len, args.vocab_limit,
         args.hidden_size, args.embedding_size, args.tweet_cov_loss_factor, args.news_cov_loss_factor,
         args.early_stopping, args.patience)

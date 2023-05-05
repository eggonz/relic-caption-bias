import sys

print(sys.version)

import torchtext
import torch
import csv
import spacy
import re
from torchtext.legacy import data
import pickle
import random
from nltk import word_tokenize
import nltk

nltk.download('punkt')
import time
import argparse
import numpy as np
import os
import pprint
from nltk.tokenize import word_tokenize
from io import open
import sys
import json
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm, trange
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import datetime


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='captioning', type=str)
    parser.add_argument("--cap_model", default='sat', type=str, help='sat, oscar, nic_plus or nic_equalizer')
    parser.add_argument("--calc_ann_leak", default=False, type=bool)
    parser.add_argument("--calc_model_leak", default=False, type=bool)
    parser.add_argument("--test_ratio", default=0.1, type=float)
    parser.add_argument("--balanced_data", default=True, type=bool)
    parser.add_argument("--mask_age_words", default=True, type=bool)
    parser.add_argument("--save_preds", default=False, type=bool)
    parser.add_argument("--use_glove", default=False, type=bool)
    parser.add_argument("--save_model_vocab", default=False, type=bool)
    parser.add_argument("--align_vocab", default=True, type=bool)
    parser.add_argument("--mask_bias_source", default='', type=str, help='obj or person or both or none')

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--save_model", default=False, type=bool)
    parser.add_argument("--workers", default=1, type=int)

    parser.add_argument("--embedding_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--pad_idx", default=0, type=int)
    parser.add_argument("--fix_length", default=False, type=bool)

    parser.add_argument("--data_dir", default='age_data', type=str)

    return parser


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text: [sent len, batch size]

        embedded = self.dropout(self.embedding(text))  # [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output: [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden), embedded


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def make_train_test_split(args, age_task_entries):
    if args.balanced_data:
        old_entries, young_entries = [], []
        for entry in age_task_entries:
            if entry['bb_age'] == 'young':
                young_entries.append(entry)
            else:
                old_entries.append(entry)
        print('young entries (before balance)', len(young_entries))
        print('old entries (before balance)', len(old_entries))
        each_test_sample_num = round(len(young_entries) * args.test_ratio)
        each_train_sample_num = len(young_entries) - each_test_sample_num

        old_train_entries = [old_entries.pop(random.randrange(len(old_entries))) for _ in
                              range(each_train_sample_num)]
        young_train_entries = [young_entries.pop(random.randrange(len(young_entries))) for _ in
                                range(each_train_sample_num)]
        old_test_entries = [old_entries.pop(random.randrange(len(old_entries))) for _ in range(each_test_sample_num)]
        young_test_entries = [young_entries.pop(random.randrange(len(young_entries))) for _ in
                               range(each_test_sample_num)]
        print('young train entries (after balance)', len(young_train_entries))
        print('old train entries (after balance)', len(old_train_entries))
        print('young test entries (after balance)', len(young_test_entries))
        print('old test entries (after balance)', len(old_test_entries))
        d_train = old_train_entries + young_train_entries
        d_test = old_test_entries + young_test_entries
        random.shuffle(d_train)
        random.shuffle(d_test)
        print('#train : #test = ', len(d_train), len(d_test))
    else:
        d_train, d_test = train_test_split(age_task_entries, test_size=args.test_ratio, random_state=args.seed,
                                           stratify=[entry['age'] for entry in age_task_entries])

    return d_train, d_test


def train(model, iterator, optimizer, criterion, train_proc):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    cnt = 0
    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.prediction

        predictions, _ = model(text, text_lengths)
        predictions = predictions.squeeze(1)
        loss = criterion(predictions, batch.label.to(torch.float32))

        acc = binary_accuracy(predictions, batch.label.to(torch.float32))

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        cnt += 1

        train_proc.append(loss.item())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), train_proc


def evaluate(model, iterator, criterion, batch_size, TEXT, args):
    calc_score = True
    calc_age_acc = True

    m = nn.Sigmoid()
    total_score = 0

    epoch_loss = 0
    epoch_acc = 0

    old_preds_all, young_preds_all = list(), list()
    old_scores_all, young_scores_all = list(), list()
    old_truth_all, young_truth_all = list(), list()
    all_pred_entries = []

    model.eval()

    with torch.no_grad():

        cnt_data = 0
        for i, batch in enumerate(iterator):

            text, text_lengths = batch.prediction

            predictions, _ = model(text, text_lengths)
            predictions = predictions.squeeze(1)
            cnt_data += predictions.size(0)

            loss = criterion(predictions, batch.label.to(torch.float32))

            acc = binary_accuracy(predictions, batch.label.to(torch.float32))

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if calc_score:
                probs = m(predictions).cpu()  # [batch_size]
                pred_ages = (probs >= 0.5000).int()

                correct = torch.eq(pred_ages, batch.label.to(torch.int32).cpu())

                pred_score_tensor = torch.zeros_like(correct, dtype=float)

                for i in range(pred_score_tensor.size(0)):
                    young_score = probs[i]
                    old_score = 1 - young_score
                    if old_score >= young_score:
                        pred_score = old_score
                    else:
                        pred_score = young_score

                    pred_score_tensor[i] = pred_score

                scores_tensor = correct.int() * pred_score_tensor
                correct_score_sum = torch.sum(scores_tensor)
                total_score += correct_score_sum.item()

            if calc_age_acc:
                probs = m(predictions).cpu()  # [batch_size]
                pred_ages = (probs >= 0.5000).int()
                old_target_ind = [i for i, x in enumerate(batch.label.to(torch.int32).cpu().numpy().tolist()) if
                                   x == 0]
                young_target_ind = [i for i, x in enumerate(batch.label.to(torch.int32).cpu().numpy().tolist()) if
                                     x == 1]
                old_pred = [*itemgetter(*old_target_ind)(pred_ages.tolist())]
                young_pred = [*itemgetter(*young_target_ind)(pred_ages.tolist())]
                old_scores = [*itemgetter(*old_target_ind)(probs.tolist())]
                old_scores = (1 - torch.tensor(old_scores)).tolist()
                young_scores = [*itemgetter(*young_target_ind)(probs.tolist())]
                old_target = [*itemgetter(*old_target_ind)(batch.label.to(torch.int32).cpu().numpy().tolist())]
                young_target = [*itemgetter(*young_target_ind)(batch.label.to(torch.int32).cpu().numpy().tolist())]
                old_preds_all += old_pred
                old_scores_all += old_scores
                old_truth_all += old_target
                young_preds_all += young_pred
                young_scores_all += young_scores
                young_truth_all += young_target

            if args.save_preds:
                probs = m(predictions).cpu()  # [batch_size]
                pred_ages = (probs >= 0.5000).int()

                for i, (imid, fs, pg) in enumerate(zip(batch.imid, probs, pred_ages)):
                    image_id = imid.item()
                    young_score = fs.item()
                    old_score = 1 - young_score

                    sent_ind = text[:, i]
                    sent_list = []
                    for ind in sent_ind:
                        word = TEXT.vocab.itos[ind]
                        sent_list.append(word)
                    sent = ' '.join([c for c in sent_list])

                    all_pred_entries.append(
                        {'image_id': image_id, 'old_score': old_score, 'young_score': young_score,
                         'input_sent': sent})

    if calc_age_acc:
        old_acc = accuracy_score(old_truth_all, old_preds_all)
        young_acc = accuracy_score(young_truth_all, young_preds_all)
        old_correct = torch.eq(torch.tensor(old_preds_all), torch.tensor(old_truth_all))
        young_correct = torch.eq(torch.tensor(young_preds_all), torch.tensor(young_truth_all))
        old_scores_tensor = old_correct.int() * torch.tensor(old_scores_all)
        old_score_sum = torch.sum(old_scores_tensor).item()
        old_score_avg = old_score_sum / (len(iterator) * batch_size * 0.5)
        young_scores_tensor = young_correct.int() * torch.tensor(young_scores_all)
        young_score_sum = torch.sum(young_scores_tensor).item()
        young_score_avg = young_score_sum / (len(iterator) * batch_size * 0.5)
    else:
        old_acc, young_acc = None, None

    if args.save_preds:
        file_name = '%s_pred_entries_seed%s.pkl' % (args.cap_model, args.seed)
        save_path = os.path.join('/bias-vl/LSTM', args.cap_model, file_name)
        pickle.dump(all_pred_entries, open(save_path, 'wb'))

    return epoch_loss / len(iterator), epoch_acc / len(
        iterator), total_score / cnt_data, old_acc, young_acc, old_score_avg, young_score_avg


def main(args):
    if os.path.exists('bias_data/train.csv'):
        os.remove('bias_data/train.csv')
    if os.path.exists('bias_data/val.csv'):
        os.remove('bias_data/val.csv')
    if os.path.exists('bias_data/test.csv'):
        os.remove('bias_data/test.csv')

    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device: {} n_gpu: {}".format(device, n_gpu))
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)

    LABEL = data.LabelField(dtype=torch.float)

    age_cap_entries = pickle.load(open(f'{args.data_dir}/age_dataset.pkl', 'rb'))

    # Select captioning model
    if args.cap_model == 'sat':
        pred_col = 'pred_sat'
    elif args.cap_model == 'oscar':
        pred_col = 'pred_oscar'
    elif args.cap_model == 'nic_plus':
        pred_col = 'pred_nicplus'
    elif args.cap_model == 'nic_equalizer':
        pred_col = 'pred_niceq'

    age_words = [
        'child', 'children', 'young', 'baby', 'babies', 'kid', 'kids', 'little', 'boy', 'boys', 'girl', 'girls',
        'old', 'elderly', 'man', 'men', 'woman', 'women', 'lady', 'ladies', 'gentleman', 'gentlemen',
        'person', 'people', 'guy', 'guys', 'teenager', 'teenagers', 'teen', 'teens', 'adult', 'adults', 'elder',
    ]

    ##################### ANN LIC score #######################
    if args.calc_ann_leak:
        print('--- calc ANN Leakage ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('-- task is Captioning --')
            d_train, d_test = make_train_test_split(args, age_cap_entries)

            val_acc_list = []
            old_acc_list, young_acc_list = [], []
            score_list = []
            old_score_list, young_score_list = [], []
            rand_score_list = []

            if args.align_vocab:
                model_vocab = pickle.load(open(f'{args.data_dir}/model_vocab/%s_vocab.pkl' % args.cap_model, 'rb'))
                print('len(model_vocab):', len(model_vocab))

            for cap_ind in range(5):
                if args.mask_age_words:
                    with open(f'{args.data_dir}/train.csv', 'w') as f:
                        writer = csv.writer(f, dialect='unix')
                        for i, entry in enumerate(d_train):
                            if entry['bb_age'] == 'old':
                                age = 0
                            else:
                                age = 1
                            ctokens = word_tokenize(entry['caption_list'][cap_ind].lower())
                            new_list = []
                            for t in ctokens:
                                if t in age_words:
                                    new_list.append('AGEWORD')
                                elif args.align_vocab:
                                    if t not in model_vocab:
                                        new_list.append('<unk>')
                                    else:
                                        new_list.append(t)
                                else:
                                    new_list.append(t)

                            new_sent = ' '.join([c for c in new_list])
                            if i <= 10 and cap_ind == 0 and args.seed == 0:
                                print(new_sent)

                            writer.writerow([new_sent.strip(), age, entry['img_id']])

                    with open(f'{args.data_dir}/test.csv', 'w') as f:
                        writer = csv.writer(f, dialect='unix')
                        for i, entry in enumerate(d_test):
                            if entry['bb_age'] == 'old':
                                age = 0
                            else:
                                age = 1
                            ctokens = word_tokenize(entry['caption_list'][cap_ind].lower())
                            new_list = []
                            for t in ctokens:
                                if t in age_words:
                                    new_list.append('AGEWORD')
                                elif args.align_vocab:
                                    if t not in model_vocab:
                                        new_list.append('<unk>')
                                    else:
                                        new_list.append(t)
                                else:
                                    new_list.append(t)

                            new_sent = ' '.join([c for c in new_list])

                            writer.writerow([new_sent.strip(), age, entry['img_id']])

                else:
                    print("!! SHOULD MASK AGE WORDS")
                    break

                nlp = spacy.load("en_core_web_sm")

                TEXT = data.Field(sequential=True, tokenize='spacy', tokenizer_language='en_core_web_sm',
                                  include_lengths=True, use_vocab=True)
                LABEL = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
                IMID = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

                train_val_fields = [
                    ('prediction', TEXT),  # process it as text
                    ('label', LABEL),  # process it as label
                    ('imid', IMID)
                ]

                train_data, test_data = torchtext.legacy.data.TabularDataset.splits(path=f'{args.data_dir}/',
                                                                                    train='train.csv', test='test.csv',
                                                                                    format='csv',
                                                                                    fields=train_val_fields)

                MAX_VOCAB_SIZE = 25000

                if args.use_glove:
                    TEXT.build_vocab(train_data, vectors="glove.6B.100d", max_size=MAX_VOCAB_SIZE)
                else:
                    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
                LABEL.build_vocab(train_data)
                print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
                print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

                train_iterator, test_iterator = data.BucketIterator.splits(
                    (train_data, test_data),
                    batch_size=args.batch_size,
                    sort_key=lambda x: len(x.prediction),  # on what attribute the text should be sorted
                    sort_within_batch=True,
                    device=device)
                INPUT_DIM = len(TEXT.vocab)
                EMBEDDING_DIM = 100
                HIDDEN_DIM = 256
                OUTPUT_DIM = 1
                N_LAYERS = 2
                BIDIRECTIONAL = True
                DROPOUT = 0.5
                PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
                # print(PAD_IDX)

                model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

                # print(f'The model has {count_parameters(model):,} trainable parameters')

                if args.use_glove:
                    pretrained_embeddings = TEXT.vocab.vectors
                    print(pretrained_embeddings.shape)
                    model.embedding.weight.data.copy_(pretrained_embeddings)

                UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
                model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
                model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

                # Training #
                optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
                criterion = nn.BCEWithLogitsLoss()

                model = model.to(device)
                criterion = criterion.to(device)

                N_EPOCHS = args.num_epochs

                best_valid_acc = float(0)

                train_proc = []
                valid_loss, valid_acc, avg_score, old_acc, young_acc, old_score_avg, young_score_avg = evaluate(
                    model, test_iterator, criterion, args.batch_size, TEXT, args)
                rand_score_list.append(avg_score)

                for epoch in range(N_EPOCHS):
                    train_loss, train_acc, train_proc = train(model, train_iterator, optimizer, criterion, train_proc)

                valid_loss, valid_acc, avg_score, old_acc, young_acc, old_score, young_score = evaluate(model,
                                                                                                            test_iterator,
                                                                                                            criterion,
                                                                                                            args.batch_size,
                                                                                                            TEXT, args)
                val_acc_list.append(valid_acc)
                old_acc_list.append(old_acc)
                young_acc_list.append(young_acc)
                score_list.append(avg_score)
                old_score_list.append(old_score)
                young_score_list.append(young_score)
                # print("Average score:", avg_score)

            young_avg_acc = sum(young_acc_list) / len(young_acc_list)
            old_avg_acc = sum(old_acc_list) / len(old_acc_list)
            avg_score = sum(score_list) / len(score_list)
            old_avg_score = sum(old_score_list) / len(old_score_list)
            young_avg_score = sum(young_score_list) / len(young_score_list)

            print('########## Results ##########')
            print(f"LIC score (LIC_D): {avg_score * 100:.2f}%")
            # print(f"\t Young score: {young_avg_score*100:.2f}%")
            # print(f"\t Old score: {old_avg_score*100:.2f}%")
            # print('!Random avg score', score_list, sum(rand_score_list) / len(rand_score_list))
            print('#############################')

    ########### MODEL LIC score ###########
    if args.calc_model_leak:
        print('--- calc MODEL Leakage ---')
        ## Captioning ##
        if args.task == 'captioning':
            print('--- task is Captioning ---')
            d_train, d_test = make_train_test_split(args, age_cap_entries)

            # !!! for qualitative !!!
            flag_imid_ = 0
            if args.mask_age_words:
                with open(f'{args.data_dir}/train.csv', 'w') as f:
                    writer = csv.writer(f, dialect='unix')
                    for i, entry in enumerate(d_train):
                        if entry['bb_age'] == 'old':
                            age = 0
                        else:
                            age = 1
                        ctokens = word_tokenize(entry[pred_col])
                        new_list = []
                        for t in ctokens:
                            if t in age_words:
                                new_list.append('AGEWORD')
                            else:
                                new_list.append(t)
                        new_sent = ' '.join([c for c in new_list])
                        if i <= 5 and args.seed == 0:
                            print(new_sent)

                        writer.writerow([new_sent.strip(), age, entry['img_id']])

                with open(f'{args.data_dir}/test.csv', 'w') as f:
                    writer = csv.writer(f, dialect='unix')
                    test_imid_list = []
                    for i, entry in enumerate(d_test):
                        test_imid_list.append(entry['img_id'])
                        if entry['bb_age'] == 'old':
                            age = 0
                        else:
                            age = 1

                        ctokens = word_tokenize(entry[pred_col])
                        new_list = []
                        for t in ctokens:
                            if t in age_words:
                                new_list.append('AGEWORD')
                            else:
                                new_list.append(t)
                        new_sent = ' '.join([c for c in new_list])

                        writer.writerow([new_sent.strip(), age, entry['img_id']])

            else:
                print("!! SHOULD MASK AGE WORDS")

        nlp = spacy.load("en_core_web_sm")

        TEXT = data.Field(sequential=True,
                          tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          include_lengths=True,
                          use_vocab=True)
        LABEL = data.Field(sequential=False,
                           use_vocab=False,
                           pad_token=None,
                           unk_token=None,
                           )
        IMID = data.Field(sequential=False,
                          use_vocab=False,
                          pad_token=None,
                          unk_token=None,
                          )

        train_val_fields = [
            ('prediction', TEXT),  # process it as text
            ('label', LABEL),  # process it as label
            ('imid', IMID)
        ]

        train_data, test_data = torchtext.legacy.data.TabularDataset.splits(path=f'{args.data_dir}/', train='train.csv',
                                                                            test='test.csv',
                                                                            format='csv', fields=train_val_fields)

        # ex = train_data[1]
        # print(ex.prediction, ex.label)

        MAX_VOCAB_SIZE = 25000

        if args.save_model_vocab:
            TEXT.build_vocab(train_data, test_data, max_size=MAX_VOCAB_SIZE)
            vocab_itos_list = TEXT.vocab.itos
            file_name = '/bias-vl/%s_vocab.pkl' % args.cap_model
            pickle.dump(vocab_itos_list, open(file_name, 'wb'))
            print('--- Saved vocab ---')

        if args.use_glove:
            print("-- Use GloVe")
            TEXT.build_vocab(train_data, vectors="glove.6B.100d", max_size=MAX_VOCAB_SIZE)
        else:
            TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
        LABEL.build_vocab(train_data)
        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
        print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
        # print(LABEL.vocab.stoi)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=args.batch_size,
            sort_key=lambda x: len(x.prediction),  # on what attribute the text should be sorted
            sort_within_batch=True,
            device=device)
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        N_LAYERS = 2
        BIDIRECTIONAL = True
        DROPOUT = 0.5
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
        # print(PAD_IDX)

        model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

        # print(f'The model has {count_parameters(model):,} trainable parameters')

        if args.use_glove:
            pretrained_embeddings = TEXT.vocab.vectors
            print(pretrained_embeddings.shape)
            model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        # Training #
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = args.num_epochs

        train_proc = []
        for epoch in range(N_EPOCHS):
            train_loss, train_acc, train_proc = train(model, train_iterator, optimizer, criterion, train_proc)

        valid_loss, valid_acc, avg_score, old_acc, young_acc, old_score, young_score = evaluate(model,
                                                                                                    test_iterator,
                                                                                                    criterion,
                                                                                                    args.batch_size,
                                                                                                    TEXT, args)
        print('########## Results ##########')
        print(f'LIC score (LIC_M): {avg_score * 100:.2f}%')
        # print(f'\t Old. score: {old_score*100:.2f}%')
        # print(f'\t Young. score: {young_score*100:.2f}%')
        print('#############################')
        print()


if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    print("---Start---")
    print('Seed:', args.seed)
    print("Epoch:", args.num_epochs)
    print("Learning rate:", args.learning_rate)
    print("Use GLoVe:", args.use_glove)
    print("Task:", args.task)
    if args.task == 'captioning' and args.calc_model_leak:
        print("Captioning model:", args.cap_model)

    if args.calc_ann_leak:
        print('Align vocab:', args.align_vocab)
        if args.align_vocab:
            print('Vocab of ', args.cap_model)
    print()

    t1 = time.time()
    try:
        main(args)
    finally:
        t2 = time.time()
        print('Elapsed time', datetime.timedelta(seconds=t2 - t1))

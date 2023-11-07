import json
import os
import copy
import re
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from dictionary import Dictionary
import csv
import collections
import nltk
import logging
import random
import numpy as np
from collections import defaultdict

def collate_fn(batch):
    max_length = 0
    for token_id, label, length in batch:
        max_length = max(length, max_length)
    token_ids = torch.zeros(len(batch), max_length, dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, (token_id, label, length) in enumerate(batch):
        token_ids[i] = torch.cat([token_id, torch.tensor([0] * (max_length - len(token_id)), dtype=torch.long)])
        labels[i] = label
    return token_ids, labels

class CLSDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', block_size=0):

        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.dictionary = dictionary
        self.padding_idx = self.dictionary.pad()
        self.vocab_size = len(dictionary)
        self.block_size = block_size
        self.max_length = 0
        self.lens = []

        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                raw_text = row[1]
                tokens = row[1].replace('\\n', ' ')
                tokens = re.sub(r'[^a-zA-Z\s]', ' ', string=tokens)
                tokens = tokens.strip().split(' ')
                
                for i in reversed(range(len(tokens))):
                    if tokens[i] == '': 
                        assert tokens.pop(i) == ''
                        continue

                    tokens[i] = tokens[i].lower()
                    if tokens[i] not in self.dictionary.indices.keys():
                        tokens[i] = '<unk>'
                token_id = self.dictionary.encode_line(tokens)
                self.max_length = max(self.max_length, len(tokens))
                self.lens.append(len(tokens))
                self.data.append((raw_text, tokens, token_id, label, len(token_id)))
        self.data = sorted(self.data, key = lambda x:x[4])
        self.idx = list(range(len(self.data)))
        if self.block_size > 0:
            l = 0
            while (l < len(self.idx)):
                r = min(len(self.idx), l + self.block_size)
                tmp = self.idx[l:r]
                random.shuffle(tmp)
                self.idx[l:r] = tmp
                l = r
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''return a batch of samples'''
        raw_text, tokens, token_id, label, length = self.data[self.idx[index]] 
        return token_id, label, length


class BOWDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', n=0):
        # n for n_gram
        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.dictionary = dictionary
        self.vocab_size = len(dictionary) 
        self.stopwords = nltk.corpus.stopwords.words('english')

        #########################################  Your Code  ###########################################
        # todo
        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                tokens = row[1].replace('\\n', ' ').lower().split()
                
                # [^a-zA-Z0-9\s.,;:!?()&@*=+#_%\^~\-"'\/\$\\]
                tokens = [token for token in tokens if token not in self.stopwords]
                tokens = [re.sub(r'[^a-z0-9]', '', string=token) for token in tokens]
                tokens = [token for token in tokens if not any(i.isdigit() for i in token)]
                
                tokens = [token if token in self.dictionary.indices.keys() else '<unk>' for token in tokens]
                token_id = self.dictionary.encode_line(tokens)
                bow_freq_feature = torch.zeros(self.vocab_size)
                for i in token_id:
                    bow_freq_feature[i] += 1
                bow_freq_feature = bow_freq_feature / len(token_id)
                self.data.append((tokens, token_id, bow_freq_feature, label))
        # raise NotImplementedError
        #################################################################################################
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # bow_freq_feature and label
        return self.data[index][-2:]


class FastTextDataset(Dataset):
    def __init__(self, data_path="./yelp_small/", dictionary=None, split='train', n=2, n_gram_size=1000):

        self.filename = os.path.join(data_path, "{}.csv".format(split))
        self.data = []
        self.dictionary = dictionary
        self.padding_idx = self.dictionary.pad()
        self.vocab_size = len(dictionary)
        self.lens = []

        #########################################  Your Code  ###########################################
        # todo
        
        def get_ngrams(word_id, n):
            ngrams = []
            for i in range(len(word_id) - n + 1):
                ngrams.append(word_id[i : i + n])
            return ngrams
        
        def hash_ngrams(ngram, vocab_size):
            base = 31
            hash = 97
            for i in ngram:
                hash = (hash ^ i) * base
            hash += hash >> 32
            return hash % vocab_size
        
        with open(self.filename) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                label = int(row[0])
                tokens = row[1].replace('\\n', ' ').lower().split()
                tokens = [token for token in tokens if token not in self.dictionary.indices.keys()]
                tokens = [re.sub(r'[^a-z0-9]', '', string=token) for token in tokens]
                tokens = [token for token in tokens if not any(i.isdigit() for i in token)]
                tokens = [token if token in self.dictionary.indices.keys() else '<unk>' for token in tokens]
                word_id = self.dictionary.encode_line(tokens)
                
                ngrams = get_ngrams(word_id, n)
                ngram_id = [hash_ngrams(ngram, n_gram_size) for ngram in ngrams]
                
                token_id = word_id + ngram_id
                
                self.lens.append(len(token_id))
                self.data.append((row[1], tokens, token_id, label, len(token_id)))

        self.total_size = self.vocab_size + n_gram_size
        # raise NotImplementedError
        #################################################################################################

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_text, tokens, token_id, label, length = self.data[index] 
        return token_id, label, length


    
    
    

        

        
        

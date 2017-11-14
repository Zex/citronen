import os
import re
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn.functional as F
from torch import nn
from torchtext import data
from subject_encoder import load_l2table

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

class SD(object):#data.Dataset):
    """
    Springer dataset
    Load and preprocess Springer text data
    Args:
    split_ratio: ratio of train and validation data set
    split_ratio = 0.7

    def __init__(self, data_path, **kwargs):
        self.data_path = data_path

        # Load data by field
        self.text_field = data.Field(lower=True)
        #self.label1_field = data.Field(sequential=False)
        self.label2_field = data.Field(sequential=False)
        self.text_field.preprocessing = data.Pipeline(clean_str)
        fields = [
            ("text", self.text_field),
        #    ("label1", label1_field),
            ("label2", self.label2_field),
        ]

        chunk = self.gen_data()
        text = chunk["desc"].tolist()
        label = chunk["subcate"].tolist()
        
        examples = [data.Example.fromlist([text[i], label[i]], fields) for i in range(len(text))]
        np.random.shuffle(examples)
        # Split train/eval data
        split_index = int(len(examples) * SD.split_ratio)
        train_data = examples[:split_index]
        eval_data = examples[split_index:]

        self.text_field.build_vocab(train_data, eval_data)
        self.label2_field.build_vocab(train_data, eval_data)
        
        self.train_data = self(self.text_field, self.label2_field, examples=train_data)
        self.eval_data = self(self.text_field, self.label2_field, examples=eval_data)

        super(SD, self).__init__(examples, fields, **kwargs)

    def gen_data(self):
        data = pd.read_csv(self.data_path, header=0, delimiter="###", engine='python')
        return data
        #self.process_chunk(data)
        # Load by chunk
        reader = pd.read_csv(self.data_path, engine='python', header=0, 
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            yield self.process_chunk(chunk)

    def process_chunk(self, chunk):
        text = chunk["desc"].tolist()
        label = chunk["subcate"].tolist()
        
        examples = [data.Example.fromlist([text[i], label[i]], self.fields) for i in range(len(text))]
        np.random.shuffle(examples)
        # Split train/eval data
        split_index = int(len(example) * SD.split_ratio)
        train_data = example[:split_index]
        eval_data = example[split_index:]
        
        self.train_data = self(self.text_field, self.label2_field, examples=train_data)
        self.eval_data = self(self.text_field, self.label2_field, examples=eval_data)
    """
    def __init__(self, args):
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.l2table = load_l2table()

    def load_data(self):
        reader = pd.read_csv(self.data_path, engine='python', header=0, 
            delimiter="###", chunksize=self.batch_size)
        for chunk in reader:
            self.process_chunk(chunk)

    def process_chunk(self, chunk):
        chunk = chunk.replace({"subcate":self.l2table})

        text = chunk["desc"]
        label = chunk["subcate"]

        np.random.seed(17)
        indices = np.random.permutation(np.arange(len(text)))
        text = text[indices].tolist()
        label = label[indices].tolist()

        text = np.narray(text)
        label = np.narray(label)

        return text, label


class Springer(nn.Module):

    def __init__(self, args):
        super(Springer, self).__init__()
        self.model_dir = args.model_dir
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.sd = SD(args)
        self.sd.load_data()
        self.global_epoch = args.global_epoch
        self.epochs = args.epochs
        self.embed_dim = 128 
        self.embed_num = len(self.sd.text_field.vocab)
        self.total_class = len(self.l2table)
        self.dropout = nn.Dropout(args.dropout)
        self.lr = args.lr

        self._build_model()

    def _build_model(self):
        self.embed = nn.Embedding(self.embed_num, self.embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, 100, (ks, self.embed_dim)) for ks in [3,4,5]]) 
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(3*100, self.total_class) 

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


def train(model):
    opt = torch.optim.Adam(model.parameters(), lr=model.lr)
    global_steps = 0
    model.train()
    
    for e in range(model.epochs):
        for batch in model.train_data:
            feat, target = batch.text, batch.label2
            feat.data.t_()
            target.data.sub_(1)
            
            opt.zero_grad()
            logit = model(feat)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            opt.step()

            global_steps += 1
            if global_steps % 1e2 == 0:
                corr = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                acc = corr/batch.batch_size * 100.0
                print("[{}] loss:{:.4f} acc:{:.4f} corr:{} bs:{}".format(
                    global_steps, loss.data[0], acc, corr, batch.batch_size), flush=True)
                if not os.path.isdir(model.model_dir):
                    os.makedirs(model.model_dir)
                output_path = "{}_steps{}.pt".format(model.model_dir, global_steps)
                torch.save(model, output_path)
                eval(model)

def eval(model):
    model.eval()
    corr, total_loss = 0, 0
    for batch in model.eval_data:
        feat, target = batch.text, batch.label2
        feat.data.t_()
        target.data.sub_(1)

        logit = model(feat)
        loss = F.cross_entropy(logit, target, size_average=False)
        total_loss += loss.data[0]
        corr += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()

    sz = total_loss/len(model.eval_data.dataset)
    loss = total_loss/sz
    acc = corr/sz * 100.0
    #model.train()
    print("[EVAL] loss:{:.4f} acc:{:.4f} corr:{} size:{}".format(
        loss.data[0], acc, corr, sz), flush=True)
    

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'validate'])
    parser.add_argument('--chunk_size', default=32, type=int, help='Load data by size of chunk')
    parser.add_argument('--data_path', default="../data/springer/mini.csv", type=str, help='Path to input data')
    parser.add_argument('--global_epoch', default=0, type=int, help='Start training from epoch')
    parser.add_argument('--epochs', default=1000, type=int, help="Total epochs to train")
    parser.add_argument('--dropout', default=0.5, type=int, help="Dropout rate")
    parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
    parser.add_argument('--batch_size', default=100, type=float, help="Batch size")
    parser.add_argument('--model_dir', default="../models/springer", type=str, help="Path to model and check point")

    args = parser.parse_args()
    return args

def start():
    args = init()
    springer = Springer(args)
    train(springer)

if __name__ == '__main__':
    start()

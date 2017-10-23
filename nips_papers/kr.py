import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3 as sql
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class WhatText(object):
    """
    CREATE TABLE papers (
        id INTEGER PRIMARY KEY,
        year INTEGER,
        title TEXT,
        event_type TEXT,
        pdf_name TEXT,
        abstract TEXT,
        paper_text TEXT);
    CREATE TABLE authors (
        id INTEGER PRIMARY KEY,
        name TEXT);
    CREATE TABLE paper_authors (
        id INTEGER PRIMARY KEY,
        paper_id INTEGER,
        author_id INTEGER);
    CREATE INDEX paperauthors_paperid_idx ON paper_authors (paper_id);
    CREATE INDEX paperauthors_authorid_idx ON paper_authors (author_id);

    """
    def __init__(self, args):
        self.data_path = args.data_path
        self.mode = args.mode
        self.chunk_size = args.chunk_size
        self.db_path = "../data/nips_papers/database.sqlite"
        self.tables = ["authors", "paper_authors", "papers"]
        self.train_rate = 0.8
        self.total_items = {}

    def train(self):
        reader = pd.read_csv(self.data_path, header=0, chunksize=self.chunk_size)
        for chunk in reader:
            self._train(chunk)

    def _train(self, chunk):
        print(chunk)

    def test(self):
        pass

    def eval(self):
        pass

    def load_db(self):
        with sql.connect(self.db_path) as conn:
            for table in self.tables:
                cur = conn.execute("select count(*) from {};".format(table))
                self.total_items[table] = cur.fetchone()[0]
                print("total {}:{}".format(table, self.total_items[table]))

            cur = cur.execute("select paper_text from papers limit {};".format(
                int(self.total_items['papers']*0.8)))
            return cur.fetchall()

    def encode_paper(self, data):
        data = [d[0] for d in data]
        cntvec = CountVectorizer()
        cntvec.fit_transform(data)
        print(cntvec.vocabulary)
        tf_matrix = cntvec.transform(data)

        tfidf = TfidfTransformer(norm="l2")
        tfidf.fit(tf_matrix)
        tfidf_matrix = tfidf.transform(tf_matrix)
        print(tfidf_matrix.shape)#.todense())

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'validate'])
    parser.add_argument('--chunk_size', default=32, type=int, help='Load data by size of chunk')
    parser.add_argument('--data_path', default=".", type=str, help='Path to input data')
    args = parser.parse_args()
    return args

def start():
    args = init()
    wt = WhatText(args)
    data = wt.load_db()
    wt.encode_paper(data)
    #wt.train()

if __name__ == '__main__':
    start()

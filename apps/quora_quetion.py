from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import scipy.sparse as sp
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter
import argparse
from os.path import isfile
import _pickle

shrink_data_source = '../data/quora/train_shrink.csv'
train_data_source = '../data/quora/train.csv'
test_data_source = '../data/quora/test.csv'
test_result = '../data/quora/test_result.csv'
model_path = "../models/quora_mlp.data"
max_feature = 366

def how_diff(cos_sim):
    angle_in_radians = math.acos(cos_sim)
    return math.degrees(angle_in_radians)

def do_cosine_sim(X, y):
    for x1, x2, y in zip(q1, q2, labels):
        tfidf_q = tfidf_vec.fit_transform(X, y)
        cos_simi = cosine_similarity(tfidf_q[:1], tfidf_q)
        print('cosine_similarity[exp:{}]: diff:{:0.2f}\n{}'.format(y, how_diff(cos_simi[0,1]), cos_simi))
    
def do_train(data, model=None):
    q1, q2, labels = data['question1'],\
                    data['question2'],\
                    data['is_duplicate']
    tfidf_vec = TfidfVectorizer(norm='l2', sublinear_tf=True, stop_words='english', max_features=max_feature)
    """
    tfidf_q1 = tfidf_vec.fit_transform(q1)
    tfidf_q2 = tfidf_vec.fit_transform(q2)
    tfidf_q = sp.hstack([tfidf_q1, tfidf_q2], format='csr')
    """
    tfidf_q = tfidf_vec.fit_transform(q1, q2)
    #print('shape: {}, {}'.format(tfidf_q.shape, labels.shape))
    #lr = LogisticRegression()
    if model is None:
        model = MLPClassifier(hidden_layer_sizes=(500, ), activation='relu', 
                        solver='adam', alpha=0.0001, batch_size='auto', 
                        learning_rate='adaptive', learning_rate_init=0.01, 
                        power_t=0.5, max_iter=1000, shuffle=True, random_state=1,
                        tol=0.0001, verbose=True, warm_start=False, momentum=0.9, 
                        nesterovs_momentum=True, early_stopping=False, 
                        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = model.fit(tfidf_q, labels)
    #print(model.loss_, model.n_outputs_, model.intercepts_)
    return model

def do_test(model, data):
    qid, q1, q2 = data['test_id'].values.astype('U'), data['question1'].values.astype('U'), data['question2'].values.astype('U')
    tfidf_vec = TfidfVectorizer(norm='l2', sublinear_tf=True, stop_words='english', max_features=max_feature)
    """
    tfidf_q1 = tfidf_vec.fit_transform(q1)
    tfidf_q2 = tfidf_vec.fit_transform(q2)
    tfidf_q = sp.hstack([tfidf_q1, tfidf_q2], format='csr')
    """
    tfidf_q = tfidf_vec.fit_transform(q1, q2)
    #score = model.score(data, targets)
    #print('score: {}'.format(score))
    return zip(qid, model.predict(tfidf_q))

def do_validate(model, data):
    q1, q2, labels = data['question1'],\
                    data['question2'],\
                    data['is_duplicate']
    tfidf_vec = TfidfVectorizer(norm='l2', sublinear_tf=True, stop_words='english', max_features=max_feature)
    tfidf_q = tfidf_vec.fit_transform(q1, q2)
    score = model.score(tfidf_q, labels)
    return score

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='Mode to run in', choices=['train', 'test', 'validate'])
    args = parser.parse_args()
    return args

def load_model(model_path):
    if isfile(model_path):
        return _pickle.load(open(model_path, 'rb'), encoding='bytes')
    return None

def start(args):
    model = load_model(model_path)
    if args.mode == 'train':
        model = train(model)
        test(model)
    elif args.mode == 'test':
        test(model)
    else:
        validate(model)

def train(model):
    # data loader
    chunksize = 100
    reader = pd.read_csv(train_data_source, header=0, chunksize=chunksize)
    model = None
    # do train
    for data in reader:
        model = do_train(data, model)
        _pickle.dump(model, open(model_path, 'wb'))

def validate(model):
    # data loader
    chunksize = 100
    reader = pd.read_csv(train_data_source, header=0, chunksize=chunksize)
    # do validate
    for data in reader:
        score = do_validate(model, data)
        print('score: {}'.format(score))

def test(model):
    if model is None:
        return
    chunksize = 500
    reader = pd.read_csv(test_data_source, header=0, chunksize=chunksize)
    # do test
    with open(test_result, 'w+') as fd:
        fd.write('"test_id","is_duplicate"\n')
        for data in reader:
            result = do_test(model, data)
            [fd.write('{},{}\n'.format(r[0], r[1])) for r in result]

if __name__ == '__main__':
    args = init()
    start(args)

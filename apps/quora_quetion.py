from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter

shrink_data_source = '../data/quora/train_shrink.csv'
train_data_source = '../data/quora/train.csv'
test_data_source = '../data/quora/test.csv'
model_path = "../models/quora_mlp.data"

def how_diff(cos_sim):
    angle_in_radians = math.acos(cos_sim)
    return math.degrees(angle_in_radians)

def do_cosine_sim(X, y):
    for x1, x2, y in zip(q1, q2, labels):
        tfidf_q = tfidf_vec.fit_transform(X, y)
        cos_simi = cosine_similarity(tfidf_q[:1], tfidf_q)
        print('cosine_similarity[exp:{}]: diff:{:0.2f}\n{}'.format(y, how_diff(cos_simi[0,1]), cos_simi))
    
def do_train(train_data):
    q1, q2, labels = train_data['question1'],\
                    train_data['question2'],\
                    train_data['is_duplicate']
    tfidf_vec = TfidfVectorizer(norm='l2', sublinear_tf=True)#, stop_words='english')
    #lr = LogisticRegression()
    mlp = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', 
                        solver='adam', alpha=0.0001, batch_size='auto', 
                        learning_rate='adaptive', learning_rate_init=0.001, 
                        power_t=0.5, max_iter=1000, shuffle=True, random_state=1,
                        tol=0.0001, verbose=True, warm_start=True, momentum=0.9, 
                        nesterovs_momentum=True, early_stopping=False, 
                        validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    tfidf_q = tfidf_vec.fit_transform(q1, q2)
    print(tfidf_q.shape, labels.shape)
    mlp = mlp.fit(tfidf_q, labels)
    return mlp

def do_test(model, test_data):
    q1, q2 = test_data['question1'], test_data['question2']
    tfidf_vec = TfidfVectorizer(norm='l2', sublinear_tf=True)
    tfidf_q = tfidf_vec.fit_transform(q1, q2)
    return model.predict(tfidf_q)



def start():
    # do train
    train_data = pd.read_csv(train_data_source, header=0)
    model = do_train(train_data)
    del train_data
    pickle.dump(model, open(model_path, 'wb'))
    # do test
    test_data = pd.read_csv(test_data_source, header=0)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    result = do_test(model, test_data)
    print('='*30, '\n', result)
start()

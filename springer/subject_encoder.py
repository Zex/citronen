import pandas as pd
import os
import pickle
import seaborn as sns

L1_TABLE_PATH = "../data/springer/l1_table.pickle"
L2_TABLE_PATH = "../data/springer/l2_table.pickle"
data_path = "../data/springer/full.csv"

l1_table = {}
l2_table = {}

def persist(obj, path):
    with open(path, 'w+b') as fd:
        pickle.dump(obj, fd)

def from_persist(path):
    with open(path, 'r+b') as fd:
        ret = pickle.load(fd)
    return ret

def encode():

    def assign_l1(cate):
        #cate = cate.replace("\\/", "/").replace("\/", "/")
        if cate not in l1_table:
            l1_table.update({cate:0x1000 if not l1_table else max(l1_table.values()) + 0x1000})

    reader = pd.read_csv(data_path, engine='python', header=0, chunksize=100, delimiter="###")
    for chunk in reader:
        chunk["cate"].apply(assign_l1)
    
    persist(l1_table, L1_TABLE_PATH)
  

    mem = {}
    def assign_l2(cate, subcate):
        #cate = cate.replace("\\/", "/").replace("\/", "/")
        #subcate = subcate.replace("\\/", "/").replace("\/", "/")
        if subcate not in l2_table:
            l2_table.update({subcate:l1_table[cate]+1 if cate not in mem else mem[cate]+1})
            mem.update({cate:l2_table[subcate]})

    reader = pd.read_csv(data_path, engine='python', header=0, chunksize=100, delimiter="###")
    for chunk in reader:
        chunk.apply(lambda x: assign_l2(x["cate"], x["subcate"]), axis=1)
    persist(l2_table, L2_TABLE_PATH)

def load_l2table():
    return from_persist(L2_TABLE_PATH)

def load_l1table():
    return from_persist(L1_TABLE_PATH)

def plot_group(data_path):
    reader = pd.read_csv(data_path, engine='python', header=0, 
        delimiter="###", chunksize=1)
    for chunk in reader:
        grp = chunk.groupby("subcate")
        #text, l1, l2 = chunk["desc"], chunk["cate"], chunk["subcate"]
        sns.boxplot(pd.DataFrame(grp.groups))

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
import nltk.data
nltk.data.path.append("/media/sf_patsnap/nltk_data")
    
def clean_lang(data_path):
    def extract_xy(chunk):
        chunk = chunk.dropna()#.replace({"subcate":self.l2table}))

        text = chunk["desc"].values[0]
        label1 = chunk["cate"].values[0]
        label2 = chunk["subcate"].values[0]

        return text, label1, label2

    reader = pd.read_csv(data_path, engine='python', header=0, 
        delimiter="###", chunksize=1)
    for chunk in reader:
        text, l1, l2 = extract_xy(chunk)
        ratio = guess_lang(text)
        lang = max(ratio, key=ratio.get)
        output = "../data/springer/lang/{}.csv".format(lang)
        if not os.path.isfile(output):
            with open(output, "w+") as fd:
                fd.write("desc###cate###subcate\n")
        else:
            with open(output, "a") as fd:
                fd.write("{}###{}###{}\n".format(text, l1, l2))

def guess_lang(text):
    ratio = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]

    for lang in stopwords.fileids():
        stopwords_set = set(stopwords.words(lang))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        ratio[lang] = len(common_elements)
   
    return ratio

if __name__ == "__main__":
    """
    encode()
    print(len(l1_table), len(l2_table))
    l2_table = load_l2table()
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    """
    data_path = "../data/springer/full.csv"
#    clean_lang(data_path)
    plot_group(data_path)

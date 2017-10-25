import pandas as pd
import pickle

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

if __name__ == "__main__":
    encode()
    print(l1_table)
    print(l2_table)
    print("#"*10)
    print(len(l1_table), len(l2_table))

    l2_table = load_l2table()
    print(len(l2_table), l2_table)

import faiss
import numpy as np
import pickle
import argparse


def index(data):
    N, D = data.shape
    print(N, D)
    nlist = 200
    p = faiss.index_factory(D, 'IVF%s,Flat' % nlist)
    p.train(data[:N // 3])
    p.add(data)
    return p

def save_index(idx, save_path):
    faiss.write_index(idx, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input')
    parser.add_argument('-o', type=str, help='output')
    args = parser.parse_args()
    x = np.load(args.filename)
    x = index(x)
    save_index(x, args.o)

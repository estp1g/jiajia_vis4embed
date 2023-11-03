import hnswlib
import numpy as np
import pickle
import argparse


def index(data: np.ndarray, space='l2'):
    """
    - data
        - 向量数据
    - space
        - 向量空间类型
        - 可选 l2, cosine 或 ip
    """
    N, D = data.shape
    ids = np.arange(N)

    # Declaring index
    p = hnswlib.Index(space = 'l2', dim = D) # possible options are l2, cosine or ip

    # Initializing index - the maximum number of elements should be known beforehand
    p.init_index(max_elements = N + 20000, ef_construction = 100, M = 8)

    p.set_ef(10)

    # Element insertion (can be called several times):
    p.add_items(data, ids)

    return p


def save_index(idx, save_path):
    idx.save_index(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='input')
    parser.add_argument('-o', type=str, help='output')
    args = parser.parse_args()
    x = np.load(args.filename)
    x = index(x)
    save_index(x, args.o)

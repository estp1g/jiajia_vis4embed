import numpy as np
from news import get_news_data
from sentence_transformers import SentenceTransformer


NEWS_PATH = 'data/2021news.parquet'
DEVICE = 'cuda:7'


if __name__ == '__main__':
    model = SentenceTransformer('models/m3e-base', device=DEVICE)

    # news to encode
    news_data = get_news_data(NEWS_PATH)
    sentences = news_data.content

    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)

    np.save('m3e-base', embeddings)
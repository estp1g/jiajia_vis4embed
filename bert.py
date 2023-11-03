from transformers import AutoTokenizer, AutoModel
from vector import Vectorize
from news import get_news_data
import numpy as np


NEWS_PATH = 'data/2021news.parquet'
DEVICE = 'cuda:0'

#模型地址
MODEL = 'models/chinese-roberta-wwm-ext'

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(DEVICE)
    news_data = get_news_data(NEWS_PATH)
    vectorize = Vectorize(tokenizer, model, model.config.max_position_embeddings ,DEVICE)
    vectors = vectorize(news_data, 32)
    np.save('bert', vectors)

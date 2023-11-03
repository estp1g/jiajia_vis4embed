import os
import numpy as np
from news import get_news_data
# os.environ['https_proxy'] = 
# os.environ['http_proxy'] = 
import openai
from tqdm import trange
import time

NEWS_PATH = 'data/2021news.parquet'
MODEL = 'text-embedding-ada-002'

if __name__ == '__main__':
    openai.api_key = ""

    # news to encode
    news_data = get_news_data(NEWS_PATH)
    sentences = news_data.content

    embeddings = []
    batch_size = 100
    for start in trange(0, len(sentences), batch_size):
        end = min(start+batch_size, len(sentences))
        mb = sentences[start: end]
        mb = [x[:min(4096, len(x))] for x in mb]
        response = openai.Embedding.create(input=mb, model=MODEL)
        embeddings.extend([d.embedding for d in response.data])
        if start % 1000 == 0:
            np.save(f'adas/{start}', np.array(embeddings))
        time.sleep(1)
    
    embeddings = np.array(embeddings)
    np.save('ada', embeddings)

import pandas as pd
import unicodedata as ucd
import pickle


NEWS_PATH = 'data/2021news.parquet'


class NewsData:
    def __init__(self, news_df, max_length=10000):
        max_length = min(max_length, len(news_df))
        news_df = news_df.iloc[:max_length]
        self.time = news_df['time'].to_list()
        self.content = news_df['content'].map(lambda x: ucd.normalize('NFKD', x)).str.replace('\t', ' ').str.replace('  ', '').to_list()
        self.headline = news_df['headline'].map(lambda x: ucd.normalize('NFKD', x)).str.replace('\t', ' ').str.replace('  ', '').to_list()

    def __getitem__(self, idx):
        return self.content[idx]
    
    def __len__(self):
        return len(self.content)


def get_news_data(news_path, max_num=10000):
    news_df = pd.read_parquet(news_path).rename(columns={'NewsContent': 'content', 'Title': 'headline', 'DeclareDate': 'time'})
    return NewsData(news_df, max_num)


if __name__ == '__main__':
    news_data = get_news_data(NEWS_PATH)
    with open('headlines.pkl', 'wb') as f:
        pickle.dump(news_data.headline, f)

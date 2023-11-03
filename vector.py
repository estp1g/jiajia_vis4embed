import numpy as np
import torch
from tqdm import trange
from news import NewsData, get_news_data


class Vectorize:
    def __init__(self, tokenizer, model, max_length, device) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.hidden_size = model.config.hidden_size
        self.max_length = max_length
        self.device = device
    
    def average_pooling(self, h):
        # h: (N, L, D)
        return h.mean(dim=1)
    
    def __call__(self, news_data: NewsData, batch_size: int) -> np.ndarray:
        hs = torch.empty(len(news_data), self.hidden_size, dtype=torch.float)
        with torch.no_grad():
            for i in trange(0, len(news_data), batch_size):
                end = min(i+batch_size, len(news_data))
                news_list = news_data[i: end]
                inputs = self.tokenizer(news_list, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)
                outputs = self.model(
                    input_ids=inputs.input_ids.to(self.device),
                    attention_mask=inputs.attention_mask.to(self.device),
                    output_hidden_states=True)
                h = outputs.hidden_states[-1]
                h = self.average_pooling(h)
                hs[i: end] = h.cpu().float()
        return hs.numpy()

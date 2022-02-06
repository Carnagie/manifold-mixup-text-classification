import torch
import numpy as np
from transformers import BertTokenizer

from twitter_utils import get_tweet_text

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# match it with the labels in q7_label from data
labels = {
    'no': 0,
    'yes': 1,
}


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        """
        self.labels = [0, 1]
        self.texts = [tokenizer(get_tweet_text(tweet_id),
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for tweet_id in df['tweet_id']]
        """
        self.labels = []
        self.texts = []
        for ind in df.index:
            try:
                tweet_text = get_tweet_text(df['tweet_id'][ind])
            except Exception as e:
                print("[ERROR]", e)
                continue
            self.labels.append(labels[df['q7_label'][ind]])
            self.texts.append(
                tokenizer(
                    tweet_text,
                    padding='max_length',
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
            )
            print(f"[ASSIGNED] [{df['tweet_id'][ind]}] [{df['q7_label'][ind]}]\n"
                  f"[TEXT] {tweet_text}")
            print("-------------------------------------------------------------")

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

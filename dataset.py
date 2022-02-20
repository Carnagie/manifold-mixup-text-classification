import torch
import numpy as np
from transformers import BertTokenizer

from twitter_utils import get_tweet_text

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# match it with the labels in q7_label from data
labels = {
    'no_not_interesting': 0,
    'yes_classified_as_in_question_6': 1,
    'yes_calls_for_action': 2,
    'yes_blame_authorities': 3,
    'yes_discusses_cure': 4,
    'yes_asks_question': 5,
    'yes_contains_advice': 6,
    'yes_discusses_action_taken': 7,
    'yes_other': 8,
    'not_sure': 9,
}


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels = []
        self.texts = []
        for ind in df.index:
            try:
                # tweet_text = get_tweet_text(df['tweet_id'][ind])
                tweet_text = df['tweet_content'][ind]
                if tweet_text == 'unusable tweet found':
                    raise Exception
            except Exception:
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

import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# match it with the labels in class_label from data also in evaluate
LABELS = {
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
    'harmful': 10,
}

INV_LABELS = {v: k for k, v in LABELS.items()}


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, model_name):
        self.labels = []
        self.texts = []
        self.tweet_ids = []
        self.raw_texts = []
        self.topic = 'covid-19'
        self.model_name = model_name
        for ind in df.index:
            try:
                # tweet_text = get_tweet_text(df['tweet_id'][ind])
                tweet_text = df['tweet_text'][ind]
                if tweet_text == 'unusable tweet found':
                    raise Exception
            except Exception:
                continue
            self.labels.append(LABELS[df['class_label'][ind]])
            self.texts.append(
                tokenizer(
                    tweet_text,
                    padding='max_length',
                    max_length=30,
                    truncation=True,
                    return_tensors="pt"
                )
            )
            self.tweet_ids.append(df['tweet_id'][ind])
            self.raw_texts.append(tweet_text)

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

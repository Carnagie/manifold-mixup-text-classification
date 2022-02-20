import numpy as np
import pandas as pd

from bert_classifier import BertClassifier
from evaluate import Evaluate
from train import Train

# data_path = 'demo_train.tsv'
data_path = 'covid19_disinfo_english_multiclass_train_included_tweets.csv'
# df = pd.read_csv(data_path, sep='\t', dtype={'tweet_id': object})
df = pd.read_csv(data_path, dtype={'tweet_id': object})
df.head()
# df = df[['tweet_id', 'q7_label']]
df = df[['tweet_content', 'q7_label']]

# split data 80:10:10
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])

EPOCHS = 5
model = BertClassifier()
LR = 1e-6

Train(model, df_train, df_val, LR, EPOCHS)

Evaluate(model, df_test)
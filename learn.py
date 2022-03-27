import numpy as np
import pandas as pd

from models.bert_classifier import BertClassifier
from models.bert_classifier_with_mixup import BertClassifierWithMixup
from evaluate import Evaluate
from train import Train


"""
change this to following:
    word_mixup
    sentence_mixup
    manifold_mixup
    bert
"""
MIX_UP_TYPE = 'sentence_mixup'

"""
# data_path = 'demo_train.tsv'
data_path = 'data/demo_train.tsv'
df = pd.read_csv(data_path, sep='\t', dtype={'tweet_id': object})
# df = pd.read_csv(data_path, dtype={'tweet_id': object})
df.head()
# df = df[['tweet_id', 'q7_label']]
df = df[['tweet_id', 'tweet_text', 'class_label']]

print(df['class_label'].value_counts())

# split data 80:10:10
# np.random.seed(112)
# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
"""

train_data_path = 'data/CT22_multilang_train.tsv'
df_train = pd.read_csv(train_data_path, sep='\t', dtype={'tweet_id': object})
df_train = df_train[['tweet_id', 'tweet_text', 'class_label']]

val_data_path = 'data/CT22_multilang_dev.tsv'
df_val = pd.read_csv(val_data_path, sep='\t', dtype={'tweet_id': object})
df_val = df_val[['tweet_id', 'tweet_text', 'class_label']]

test_data_path = 'data/CT22_multilang_dev_test.tsv'
df_test = pd.read_csv(test_data_path, sep='\t', dtype={'tweet_id': object})
df_test = df_test[['tweet_id', 'tweet_text', 'class_label']]


print('train\n', df_train['class_label'].value_counts())
print('val\n', df_val['class_label'].value_counts())
print('test\n', df_test['class_label'].value_counts())


EPOCHS = 1

if MIX_UP_TYPE == 'manifold_mixup':
    model = BertClassifierWithMixup()
    model_name = MIX_UP_TYPE
else:
    model = BertClassifier()
    model_name = 'bert-base-case'

LR = 1e-6


Train(model, df_train, df_val, LR, EPOCHS, mix_up_type=MIX_UP_TYPE, model_name=model_name)

Evaluate(model, df_test, model_name=model_name)

import numpy as np
import pandas as pd
from transformers import BertTokenizer
from twitter_utils import get_tweet_text

# Load data
data_path = 'data/covid19_disinfo_english_binary_train.tsv'
df = pd.read_csv(data_path, sep='\t', dtype={'tweet_id': object})
df.head()
df = df[['tweet_id', 'q7_label']]

print("data:")
print(df)

test_entry = {"tweet_id": df.iloc[0][0], "q7_label": df.iloc[0][1]}

"""
# tokenizer = BertTokenizer.from_pretrained('bert-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

labels = {
    "yes": 1,
    "no": 0,
}

# pad each sequence to maximum length
# truncation if maximum length exceeded
# type of tensor (pt for pytorch) (tf for tensorflow)
bert_input = tokenizer(
    get_tweet_text(test_entry['tweet_id']),
    padding='max_length',
    truncation=True,
    return_tensors='pt',
)
"""

"""
print("Example Decoded:", tokenizer.decode(bert_input.input_ids[0]))
print(bert_input['input_ids'])
print(bert_input['token_type_ids'])
print(bert_input['attention_mask'])
"""

# split data 80:10:10
np.random.seed(112)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])

print(len(df_train), len(df_val), len(df_test))


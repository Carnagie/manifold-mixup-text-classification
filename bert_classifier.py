from torch import nn
from transformers import BertModel

"""
We then pass the pooled_output variable into a linear layer with ReLU activation function. 
At the end of the linear layer, we have a vector of size 5, each corresponds to a category of our labels
currently (yes, no)
"""


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        # embedding vectors of all sequence & embedding vectors of all CLS token
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

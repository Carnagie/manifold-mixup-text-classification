from torch import nn
from transformers import BertModel
from data_augmenters import mix_up
from data_augmenters import MIX_UP_PROB_LENGTH

"""
We then pass the pooled_output variable into a linear layer with ReLU activation function. 
At the end of the linear layer, we have a vector of size 5, each corresponds to a category of our labels
currently (yes, no)
"""


class BertClassifierWithMixup(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifierWithMixup, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 11)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask, mix_up_dict=None):
        # embedding vectors of all sequence & embedding vectors of all CLS token
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)

        if mix_up_dict:
            pooled_output = mix_up(pooled_output, 0, mix_up_dict, self.training)

        dropout_output = self.dropout(pooled_output)
        
        if mix_up_dict:
            dropout_output = mix_up(dropout_output, 1, mix_up_dict, self.training)

        linear_output = self.linear(dropout_output)

        if mix_up_dict:
            linear_output = mix_up(linear_output, MIX_UP_PROB_LENGTH - 1, mix_up_dict, self.training)

        final_layer = self.sigmoid(linear_output)

        return final_layer

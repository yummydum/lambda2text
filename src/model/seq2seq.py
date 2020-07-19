from transformers import AlbertModel
from transformers.modeling_bert import ACT2FN
import torch
from torch import nn


class AlbertDecoder(nn.Module):
    """
    Map hidden state to natural language
    Refer to AlbertMLMHead 
    """
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        import ipdb
        ipdb.set_trace()

        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        prediction_score = self.decoder(hidden_states)
        return prediction_score


class AlbertSeq2Seq(nn.Module):
    def __init__(self, config1, config2):
        super().__init__()
        # assert config1.hidd == ??
        self.encoder = AlbertModel(config1)  # formal representaion ->  hidden
        self.decoder = AlbertDecoder(config2)  # hidden -> natural language

    def forward(self, input_ids):
        outputs = self.encoder(input_ids=input_ids)
        sequence_outputs = outputs[0]
        prediction_scores = self.decoder(sequence_outputs)
        return prediction_scores

    def decoder_mask():
        return 
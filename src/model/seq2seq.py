from transformers import AlbertModel, AlbertPreTrainedModel
from transformers.modeling_bert import ACT2FN
import torch
from torch import nn


class AlbertDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        prediction_score = self.decoder(hidden_states)
        return prediction_score


class AlbertSeq2Seq(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = AlbertModel(config)
        self.decoder = AlbertDecoder(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.decoder.decoder,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        sequence_outputs = outputs[0]
        prediction_scores = self.decoder(sequence_outputs)
        return prediction_scores
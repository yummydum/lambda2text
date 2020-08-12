import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, device, dropout):
        super().__init__()

        self.name = 'lstm'
        self.src_pad_idx = 1
        self.encoder = Encoder(input_dim, hid_dim, hid_dim, hid_dim, dropout)
        self.decoder = Decoder(output_dim, hid_dim, hid_dim, hid_dim, dropout)
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=1):

        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        #first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.create_mask(src)

        #mask = [batch size, src len]

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs,
                                             mask)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, enc_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):

        #src = [src len, batch size]
        #src_len = [batch size]

        embedded = self.dropout(self.embedding(src))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, src_len, enforce_sorted=False)

        packed_outputs, hidden = self.rnn(packed_embedded)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):

        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.attention = Attention(enc_hid_dim, dec_hid_dim)

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(enc_hid_dim + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim,
                                output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):

        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch size, emb dim]
        a = self.attention(hidden[0].squeeze(0), encoder_outputs, mask)

        #a = [batch size, src len]

        a = a.unsqueeze(1)

        #a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        #weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        #weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden)

        assert (output == hidden[0]).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded),
                                           dim=1))

        #prediction = [batch size, output dim]

        return prediction, hidden, a.squeeze(1)
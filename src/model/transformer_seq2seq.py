import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(
            self,
            input_dim,  # ntokens of input
            output_dim,  # ntokens of output
            hid_dim,  # dim of word embed, encoder hid, decoder hid
            n_heads,
            n_layers,
            dropout,
            device,
            pf_dim=512,
            max_len=100):
        super(Seq2Seq, self).__init__()

        # Settings
        self.name = 'transformer'
        self.pad_idx = 1
        self.device = device

        assert hid_dim % 2 == 0  # for positional encoder

        self.encoder = Encoder(input_dim=input_dim,
                               hid_dim=hid_dim,
                               n_layers=n_layers,
                               pf_dim=pf_dim,
                               dropout=dropout,
                               n_heads=n_heads,
                               device=self.device,
                               max_len=max_len)
        self.decoder = Decoder(output_dim=output_dim,
                               hid_dim=hid_dim,
                               n_layers=n_layers,
                               n_heads=n_heads,
                               pf_dim=pf_dim,
                               dropout=dropout,
                               device=self.device,
                               max_len=max_len)

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        ones = torch.ones((trg_len, trg_len)).to(trg.device)
        trg_sub_mask = torch.tril(ones).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, tgt):

        # Make masks beforehand
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_trg_mask(tgt)

        # Forward
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, src_mask)
        return output


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout,
                 device, max_len):
        super().__init__()

        assert n_layers > 0

        self.hid_dim = hid_dim
        self.tok_embed = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)

        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, src, src_mask):
        batch_size = src.size()[0]
        src_len = src.size()[1]
        scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(src.device)
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size,
                                                           1).to(src.device)
        x = self.dropout((self.tok_embed(src) * scale) +
                         self.pos_embedding(pos))

        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads,
                                                      dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    # Returns [batch size, src len, hid dim]
    def forward(self, src, src_mask):

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        # [batch_size, src_len, hid_dim]

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout,
                 device, max_len):
        super().__init__()

        assert n_layers > 0

        self.hid_dim = hid_dim
        self.tok_embed = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_len, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
            for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device

    # Returns [batch size, trg len, output dim]
    def forward(self, trg, enc_src, trg_mask, src_mask):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size,
                                                           1).to(self.device)
        scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(trg.device)
        x = self.dropout((self.tok_embed(trg) * scale) +
                         self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(x, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads,
                                                      dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    # Returns [batch size, trg len, hid dim]
    def forward(self, trg, enc_src, trg_mask, src_mask):

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # [batch_size, target_len, hid_dim]

        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src,
                                                 src_mask)

        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg, attention


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads,
                   self.head_dim).permute(0, 2, 1, 3)

        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(query.device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)  # why drop out attention?
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

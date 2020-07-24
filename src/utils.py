import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tokenizers import SentencePieceBPETokenizer
from transformers.optimization import AdamW

from config import DATA_DIR

tokenizer_dir = DATA_DIR / 'tokenizers/SentencePiece'
TOKENIZER = SentencePieceBPETokenizer(str(tokenizer_dir / 'vocab.json'),
                                      str(tokenizer_dir / 'merges.txt'))


def get_optimzer(model, lr, decay=0.0):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def tokenize_formal(line):
    """
    Tokenize by BPE
    """
    result = TOKENIZER.encode(line).tokens
    return ''.join(result).split('▁')


def translate_sentence(formula,
                       src_field,
                       trg_field,
                       model,
                       device,
                       max_len=50):

    model.eval()

    assert hasattr(src_field, 'vocab'), 'build vocab first!'

    # Encode
    tokens = tokenize_formal(formula)
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # Decode step by step
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for _ in range(max_len):

        # Decoder forward
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask,
                                              src_mask)

        # max logit for the last element
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        # Break if end of sentence pridicted
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # Convert to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


def display_attention(sentence,
                      translation,
                      attention,
                      n_heads=8,
                      n_rows=4,
                      n_cols=2):

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=4)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower()
                                               for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
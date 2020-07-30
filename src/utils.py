import joblib
import torch
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from torchtext.data.metrics import bleu_score
from transformers.optimization import AdamW
from config import DATA_DIR

SRC_TOKENIZER = joblib.load(DATA_DIR / 'tokenizers/tokenizer_formal.joblib')
TRG_TOKENIZER = joblib.load(DATA_DIR / 'tokenizers/tokenizer_text.joblib')


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
    result = SRC_TOKENIZER.encode(line).tokens
    return result


def tokenize_text(line):
    result = TRG_TOKENIZER.encode(line).tokens
    return result


def translation_example(data, model, src_field, trg_field, device):
    for i, example in enumerate(data):

        print('Original sentence is:')
        original = example.text[0].squeeze().tolist()
        original = ' '.join([src_field.vocab.itos[x] for x in original][1:-1])
        print(original)

        print('Formal representation is:')
        formula = example.formal[0].squeeze().tolist()
        formula = ' '.join([trg_field.vocab.itos[x] for x in formula][1:-1])
        print(formula)

        print('Translation result is:')
        result, _ = translate_sentence(formula, trg_field, src_field, model,
                                       device)
        print(result)

        if i == 10:
            break
    return


def translate_sentence(src,
                       src_field,
                       trg_field,
                       model,
                       device,
                       max_len=50,
                       formula=True):

    model.eval()

    assert hasattr(src_field, 'vocab'), 'build vocab first!'

    # Encode
    if isinstance(src, str):
        if formula:
            tokens = tokenize_formal(src)
        else:
            # pesky import
            from preprocess.dataset_multi30k import tokenize_de
            tokens = [x.lower() for x in tokenize_de(src)]
    elif isinstance(src, list):
        tokens = src
    else:
        raise ValueError()
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


def calculate_bleu(data,
                   src_field,
                   trg_field,
                   model,
                   device,
                   max_len=50,
                   trans_path=None,
                   formula=True,
                   limit=None):

    trgs = []
    pred_trgs = []

    if trans_path is not None:
        f = trans_path.open(mode='w')

    count = 0
    for datum in data:

        if limit is not None and count > limit:
            break

        src = vars(datum)['src'][0][0]
        trg = vars(datum)['trg'][0][0]

        src = [src_field.vocab.itos[x] for x in src]
        trg = [trg_field.vocab.itos[x] for x in trg]

        pred_trg, _ = translate_sentence(src, src_field, trg_field, model,
                                         device, max_len, formula)

        if trans_path is not None:
            f.write(' '.join(src) + '\n')
            f.write(' '.join(trg) + '\n')
            f.write(' '.join(pred_trg) + '\n')
            f.write('\n')

        #cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    if trans_path is not None:
        f.close()

    return bleu_score(pred_trgs, trgs)


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
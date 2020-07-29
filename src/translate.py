import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from preprocess.dataset import load_datasets, FORMAL, TEXT
from utils import tokenize_formal
from config import DATA_DIR

model_dir = DATA_DIR / 'trained_model'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('id')
    parser.add_argument('--test_run')
    args = parser.parse_args()
    return args

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


def main():
    """
    python translate.py 
    """

    args = set_args()

    # Load dataset
    print('Now loading datasets...')
    train,dev,test = load_datasets(1,DEVICE)

    # Load trained model
    print('Now loading model...')
    model_path = model_dir / f'{args.id}.pt'
    model = torch.load(model_path).module

    # Show examples for test dataset
    print('Now translating...')
    result_path = Path(DATA_DIR / 'translation.txt')
    count = 0
    with result_path.open(mode='w',encoding='utf-8') as f:
        for example in tqdm(train):
            
            golden = example.text[0].squeeze().tolist()
            golden = ' '.join([TEXT.vocab.itos[x] for x in golden][1:-1]) 
            f.write(golden + '\n') 

            inputs = example.formal[0].squeeze().tolist()
            print(inputs)
            inputs = ' '.join([FORMAL.vocab.itos[x] for x in inputs][1:-1])
            f.write(inputs + '\n')

            result,_ = translate_sentence(inputs, FORMAL, TEXT, model, DEVICE)
            f.write(' '.join(result)+ '\n\n')
            count += 1
        
            if count > 100:
                break
    # Additional interactive session
    # print('You can type something now')
    # while True:
    #     src = input()
    #     result,_ = translate_sentence(src, FORMAL, TEXT, model, DEVICE)
    #     print(result)

    return


if __name__ == "__main__":
    main()
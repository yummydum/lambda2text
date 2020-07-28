import argparse
from pathlib import Path
from spacy import load
import torch
from preprocess.dataset import load_datasets, TEXT, FORMAL
from utils import translate_sentence
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

def main():
    """
    python translate.py 
    """

    args = set_args()

    # Load dataset
    print('Now loading datasets...')
    _, _, test = load_datasets(1, DEVICE)

    # Load trained model
    print('Now loading model...')
    model_path = model_dir / f'{args.id}.pt'
    model = torch.load(model_path).module

    # Show examples for test dataset
    for i, example in enumerate(test):
        
        print('Original sentence is:')
        original = example.text[0].squeeze().tolist()
        original = ' '.join([TEXT.vocab.itos[x] for x in original][1:-1]) 
        print(original)

        print('Formal representation is:')
        formula = example.formal[0].squeeze().tolist()
        formula = ' '.join([FORMAL.vocab.itos[x] for x in formula][1:-1])
        print(formula)

        print('Translation result is:')
        result,_ = translate_sentence(formula, FORMAL, TEXT, model, DEVICE)
        print(result)

        if i == 3:
            break

    # Additional interactive session
    while True:
        src = input()
        result,_ = translate_sentence(src, FORMAL, TEXT, model, DEVICE)
        print(result)

    return


if __name__ == "__main__":
    main()
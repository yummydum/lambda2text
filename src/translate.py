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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('id')
    args = parser.parse_args()

    # Load dataset
    _, _, test = load_datasets(1, 'cpu')

    # Load trained model
    model_path = model_dir / f'{args.id}.pt'
    model = torch.load(model_path)

    # Show examples for test dataset
    for i, example in enumerate(test):
        formula = example.formal[0].squeeze().tolist()
        formula = ' '.join([FORMAL.vocab.itos[x] for x in formula][1:-1])
        result = translate_sentence(formula, FORMAL, TEXT, model, DEVICE)
        print(result)
        if i == 9:
            break

    # Additional interactive session
    while True:
        src = input()
        result = translate_sentence(src, FORMAL, TEXT, model, DEVICE)
        print(result)

    return


if __name__ == "__main__":
    main()
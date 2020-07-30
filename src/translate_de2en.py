import argparse
from pathlib import Path
import torch
from tqdm import tqdm
from preprocess.dataset_multi30k import load_Multi30k, ENGLISH, GERMAN, tokenize_de, tokenize_en
from utils import translate_sentence_ge2en
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
    # _, _, test = load_datasets(1, DEVICE)
    train, dev, test = load_Multi30k(1, DEVICE)

    # Load trained model
    print('Now loading model...')
    model_path = model_dir / f'{args.id}.pt'
    model = torch.load(model_path).module

    # Show examples for test dataset
    print('Now translating...')
    result_path = Path(DATA_DIR / 'translation_de2en.txt')
    count = 0
    with result_path.open(mode='w', encoding='utf-8') as f:
        for example in tqdm(train):

            golden = example.trg[0].squeeze().tolist()
            golden = ' '.join([ENGLISH.vocab.itos[x] for x in golden][1:-1])
            f.write(golden + '\n')

            inputs = example.src[0].squeeze().tolist()
            print(inputs)
            inputs = ' '.join([GERMAN.vocab.itos[x] for x in inputs][1:-1])
            f.write(inputs + '\n')

            result, _ = translate_sentence(inputs, GERMAN, ENGLISH, model,
                                           DEVICE)
            f.write(' '.join(result) + '\n\n')
            count += 1

            if count > 100:
                break

    # Additional interactive session
    # print('You can type something now')
    # while True:
    #     src = input()
    #     result,_ = translate_sentence(src, SRC, TRG, model, DEVICE)
    #     print(result)

    return


if __name__ == "__main__":
    main()
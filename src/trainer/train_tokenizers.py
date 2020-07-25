import joblib
import pandas as pd
from tokenizers import SentencePieceBPETokenizer
from config import DATA_DIR


def main():
    result_dir = DATA_DIR / 'tokenizers'
    if not result_dir.exists():
        result_dir.mkdir()

    # Source
    mnli_path = DATA_DIR / 'pairs/mnli.tsv'
    df = pd.read_csv(mnli_path, sep='\t')

    for x in ['formal','text']:
        train_data_path = result_dir / f'train_data_{x}.txt'
        df[x].to_csv(train_data_path, header=False, index=False)

        # Train
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train([str(train_data_path)],vocab_size=20000)

        # Save
        tokenizer_path = result_dir / f'tokenizer_{x}.joblib'
        joblib.dump(tokenizer,tokenizer_path)


if __name__ == '__main__':
    main()

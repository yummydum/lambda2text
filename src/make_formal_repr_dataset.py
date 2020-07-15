from pathlib import Path
import re
from transformers import AlbertTokenizer

data_path = Path('../data/formal')
TOKENIZER = AlbertTokenizer.from_pretrained('albert-base-v2')


def main():
    for source in data_path.iterdir():
        text_path = Path(str(source).replace('formal', 'glue'))
        result_path = Path(
            str(source).replace('formal', 'pairs').replace('.txt', '.tsv'))

        with source.open(mode='r') as f_r:
            with text_path.open(mode='r') as f_r2:
                with result_path.open(mode='w') as f_w:
                    f_w.write('formal\tencoded\n')
                    for line in f_r:
                        if line.startswith('ID='):
                            continue
                        elif line == 'FAILED!\n':
                            next(f_r2)
                            continue

                        text = next(f_r2)
                        encoded = TOKENIZER.encode(text)
                        encoded = ' '.join([str(i) for i in encoded])

                        # add space for tokenization
                        line = line.replace('(', ' ( ').replace(')', ' ) ')

                        # normalize variable num
                        for c in ['x', 'e', 'F', 'DOT']:
                            variables = re.findall(f'{c}' + r'\d+', line)
                            variables = sorted(list(set(variables)))
                            for i, v in enumerate(variables):
                                line = line.replace(v, f' {c}{i} ')

                        # Tokenized
                        formal = ' '.join(line.split())
                        f_w.write(f'{formal}\t{encoded}\n')


if __name__ == "__main__":
    main()
from pathlib import Path
import re

data_path = Path('../data/formal')


def main():

    token2id = dict()

    for source in data_path.iterdir():
        tokenized_path = Path(str(source).replace('formal', 'formal_cleaned'))
        encoded_path = Path(str(source).replace('formal', 'formal_encoded'))
        with source.open(mode='r') as f_r:
            with tokenized_path.open(mode='w') as f_w:
                with encoded_path.open(mode='w') as f_w2:
                    for line in f_r:

                        if line.startswith('ID=') or line == 'FAILED!\n':
                            continue

                        # add space for tokenization
                        line = line.replace('(', ' ( ').replace(')', ' ) ')

                        # normalize variable num
                        for c in ['x', 'e']:
                            variables = re.findall(f'{c}' + r'\d+', line)
                            variables = sorted(list(set(variables)))
                            for i, v in enumerate(variables):
                                line = line.replace(v, f' {c}{i} ')

                        # Tokenized
                        tokens = line.split()
                        f_w.write(' '.join(tokens) + '\n')

                        # Encoded
                        encoded = []
                        for t in tokens:
                            if t not in token2id:
                                token2id[t] = len(token2id.keys())
                            encoded.append(token2id[t])
                        f_w2.write(' '.join(encoded) + '\n')

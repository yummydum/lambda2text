from pathlib import Path
import re
from config import DATA_DIR

data_path = DATA_DIR / 'formal'
trans_path = DATA_DIR / 'translation'
src_train_path = trans_path / 'src-train.txt'
tgt_train_path = trans_path / 'tgt-train.txt'
src_val_path = trans_path / 'src-val.txt'
tgt_val_path = trans_path / 'tgt-val.txt'


def tokenize_formal(line):
    # add space for tokenization
    line = line.replace('(', ' ( ').replace(')', ' ) ')

    # normalize variable num
    for c in ['x', 'e', 'F', 'DOT']:
        variables = re.findall(f'{c}' + r'\d+', line)
        variables = sorted(list(set(variables)))
        for i, v in enumerate(variables):
            line = line.replace(v, f' {c}{i} ')

    # cut redundant space
    line = ' '.join(line.split())
    return line


def main():
    with src_train_path.open(mode='w') as f_src:
        with tgt_train_path.open(mode='w') as f_tgt:
            with src_val_path.open(mode='w') as f_src_v:
                with tgt_val_path.open(mode='w') as f_tgt_v:
                    count = 0
                    for source in data_path.iterdir():
                        text_path = Path(str(source).replace('formal', 'glue'))
                        with source.open(mode='r') as f_r:
                            with text_path.open(mode='r') as f_r2:
                                for line in f_r:

                                    # Skip cases
                                    if line.startswith('ID='):
                                        continue
                                    elif line == 'FAILED!\n':
                                        next(f_r2)
                                        continue

                                    count += 1

                                    # Write to file
                                    formal = tokenize_formal(line)
                                    text = next(f_r2)
                                    if count % 10 != 0:
                                        f_src.write(formal + '\n')
                                        f_tgt.write(text)
                                    else:
                                        f_src_v.write(formal + '\n')
                                        f_tgt_v.write(text)

    print(count)


if __name__ == "__main__":
    main()
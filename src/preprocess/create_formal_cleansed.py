import re
import csv
from config import DATA_DIR


def normalize(line):
    """
    Rule based tokenizer
    """
    # TODO: currently has issue for COMMA and DOT

    # add space for tokenization
    line = line.replace('(', ' ( ').replace(')', ' ) ')

    # normalize variable num
    for c in ['x', 'e', 'F', 'DOT', 'z']:
        variables = re.findall(f'{c}' + r'\d+', line)
        variables = sorted(list(set(variables)))
        for i, v in enumerate(variables):
            line = line.replace(v, f' {c}{i} ')

    # cut redundant space
    line = line.split()

    # remove heading underscore
    line = [x.lstrip('_') for x in line]
    return line


def main():
    formal_split = DATA_DIR / 'formal_split'
    for source in formal_split.iterdir():

        text_path = DATA_DIR / 'mnli_split' / source.name
        result_path = DATA_DIR / 'formal_cleansed/mnli.tsv'

        with source.open(mode='r') as f_r:
            with text_path.open(mode='r') as f_r2:
                with result_path.open(mode='w') as f_w:

                    writer = csv.writer(f_w,
                                        lineterminator='\n',
                                        delimiter='\t')
                    writer.writerow(['formal', 'text'])

                    for line in f_r:

                        # Skip unneccesary lines
                        if line.startswith('ID='):
                            continue
                        elif line == 'FAILED!\n':
                            next(f_r2)
                            continue

                        # preprocess & write
                        text = next(f_r2)
                        formal = normalize(line)
                        assert '\t' not in formal
                        assert '\t' not in text
                        writer.writerow([formal, text])


if __name__ == "__main__":
    main()
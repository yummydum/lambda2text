from pathlib import Path
from config import DATA_DIR

data_path = DATA_DIR / 'formal'


def main():
    for source in data_path.iterdir():
        text_path = Path(str(source).replace('formal', 'glue'))
        result_path = Path(
            str(source).replace('formal', 'pairs').replace('.txt', '.tsv'))

        with source.open(mode='r') as f_r:
            with text_path.open(mode='r') as f_r2:
                with result_path.open(mode='w') as f_w:
                    f_w.write('formal\ttext\n')
                    for line in f_r:
                        if line.startswith('ID='):
                            continue
                        elif line == 'FAILED!\n':
                            next(f_r2)
                            continue

                        text = next(f_r2)
                        formal = ' '.join(line.split())
                        f_w.write(f'{formal}\t{text}')


if __name__ == "__main__":
    main()
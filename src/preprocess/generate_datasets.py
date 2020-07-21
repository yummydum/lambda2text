import csv
from config import DATA_DIR

data_path = DATA_DIR / 'formal'


def main():
    for source in data_path.iterdir():
        text_path = DATA_DIR / 'glue' / source.name
        result_path = DATA_DIR / 'pairs' / source.name.replace('.txt', '.tsv')

        with source.open(mode='r') as f_r:
            with text_path.open(mode='r') as f_r2:
                with result_path.open(mode='w') as f_w:

                    writer = csv.writer(f_w,
                                        lineterminator='\n',
                                        delimiter='\t')
                    writer.writerow(['formal', 'text'])

                    for line in f_r:
                        if line.startswith('ID='):
                            continue
                        elif line == 'FAILED!\n':
                            next(f_r2)
                            continue

                        text = next(f_r2)
                        formal = ' '.join(line.split())

                        assert '\t' not in formal
                        assert '\t' not in text

                        writer.writerow([formal, text])
                        # if 'x043135' in formal:
                        #     breakpoint()


if __name__ == "__main__":
    main()
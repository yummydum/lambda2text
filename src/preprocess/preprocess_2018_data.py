import csv
from config import DATA_DIR

source_dir = DATA_DIR / 'snli_datasets'
cleaned_dir = DATA_DIR / 'snli_datasets_cleansed'

if not cleaned_dir.exists():
    cleaned_dir.mkdir()

with (source_dir / 'snli_0428.txt').open('r') as f_r:
    with (cleaned_dir / 'snli_simple.tsv').open('w') as f_w:
        writer = csv.writer(f_w, delimiter='\t')
        writer.writerow(['src', 'original', 'trg'])
        for line in f_r:
            line = line.replace('\n', '').replace('(',
                                                  ' ( ').replace(')', ' ) ')
            line = ' '.join(line.split()).rstrip()
            line = line.split('# ')
            writer.writerow(line)

with (source_dir / 'snli_0428_formula.txt').open('r') as f_r:
    with (cleaned_dir / 'snli_formula.tsv').open('w') as f_w:
        writer = csv.writer(f_w, delimiter='\t')
        writer.writerow(['src', 'original', 'trg'])
        for line in f_r:
            line = line.replace('\n', '').replace('(',
                                                  ' ( ').replace(')', ' ) ')
            line = ' '.join(line.split()).rstrip()
            line = line.split('# ')
            writer.writerow(line)

with (source_dir / 'snli_0428_graph.txt').open('r') as f_r:
    with (cleaned_dir / 'snli_graph.tsv').open('w') as f_w:
        writer = csv.writer(f_w, delimiter='\t')
        writer.writerow(['src', 'original', 'trg'])
        for line in f_r:
            line = line.replace(',', ' , ').replace('\n', '')
            line = line.split('# ')
            writer.writerow(line)

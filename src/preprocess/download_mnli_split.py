from tqdm import tqdm
from nlp import load_dataset
from config import DATA_DIR

f_size = 100
data_path = DATA_DIR / 'mnli_split'
if not data_path.exists():
    data_path.mkdir()
dataset = load_dataset('glue', 'mnli')['train']
result_path = data_path / f'mnli_0.txt'
count = 0
f = result_path.open(mode='w')
for pair in tqdm(dataset):
    key1 = 'premise'
    key2 = 'hypothesis'

    for key in [key1, key2]:
        sentence = ' '.join(pair[key].strip().split())
        f.write(sentence + '\n')

    count += 1
    if count > 0 and count % f_size == 0:
        f.close()
        result_path = data_path / f'mnli_{count // f_size}.txt'
        f = result_path.open(mode='w')

f.close()

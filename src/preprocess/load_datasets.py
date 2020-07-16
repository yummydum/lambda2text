from pathlib import Path
from tqdm import tqdm
from nlp import load_dataset
import spacy
from config import DATA_DIR

data_path = DATA_DIR / 'glue'
nlp = spacy.load("en_core_web_sm")

# 'qqp','qnli' skipped since question format necessary
for data in ['mrpc', 'mnli', 'rte', 'wnli']:
    dataset = load_dataset('glue', data)['train']
    result_path = data_path / f'{data}_0.txt'
    count = 0
    f = result_path.open(mode='w')
    for pair in tqdm(dataset):

        if data in {'mrpc', 'wnli', 'rte'}:
            key1 = 'sentence1'
            key2 = 'sentence2'
        elif data in {'qqp'}:
            key1 = 'question1'
            key2 = 'question2'
        elif data in {'mnli'}:
            key1 = 'premise'
            key2 = 'hypothesis'
        else:
            raise ValueError()

        for key in [key1, key2]:
            sentences = ' '.join(pair[key].strip().split())
            doc = nlp(sentences)
            for sent in doc.sents:
                # filter too long or question
                if len(sent.text.split()) <= 60 and '?' not in sent.text:
                    f.write(sent.text)
            f.write('\n')  # preserve original pair
        count += 1
        if count // 10000 > 0:
            f.close()
            result_path = data_path / f'{data}_{count // 10000}.txt'
            f = result_path.open(mode='w')

    f.close()

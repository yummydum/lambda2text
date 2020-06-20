from pathlib import Path
import re

data_path = Path('../data/formal')

for source in data_path.iterdir():
    result_path = Path(str(source).replace('formal', 'formal_cleaned'))
    with source.open(mode='r') as f_r:
        with result_path.open(mode='w') as f_w:
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

                line = ' '.join(line.split())
                f_w.write(line + '\n')

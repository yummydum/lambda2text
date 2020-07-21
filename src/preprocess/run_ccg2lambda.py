import logging
import subprocess
from tqdm import tqdm
from config import DATA_DIR

logging.basicConfig(level=logging.DEBUG)

glue_split = DATA_DIR / 'glue_split'
formal_spilt = DATA_DIR / 'formal_split'

mrpc_num = 4
mnli_num = 370
rte_num = 4
wnli_num = 5

process_list = []
for i in tqdm(range(mnli_num + 1)):
    fn = formal_spilt / f'mnli_{i}.txt'

    # May skip if the result is already present
    if fn.exists():
        content = fn.read_text().split('\n')
        if len(content) > 100:
            continue
        else:
            pass

    logging.info(f'Now running {fn}')
    # cmd = ['make', 'ccg2lambda', f'file={fn.name}']
    cmd = ['make', 'ccg2lambda', f'file={fn.name}', 'gpu=0']
    process_list.append(subprocess.Popen(cmd, stdout=subprocess.PIPE))

    if len(process_list) > 3:
        for p in process_list:
            p.wait()

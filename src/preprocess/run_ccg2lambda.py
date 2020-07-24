import os
import signal
import time
import logging
import subprocess
from tqdm import tqdm
from config import DATA_DIR, ROOT_DIR

# Run make from root dir
os.chdir(ROOT_DIR)

logging.basicConfig(level=logging.DEBUG)

mnli_split = DATA_DIR / 'mnli_split'
formal_spilt = DATA_DIR / 'formal_split'

job_num = 6


def main():
    process_list = []
    try:
        for f_path in tqdm(sorted(mnli_split.iterdir())):

            # May skip if the result is already present
            result_path = formal_spilt / f_path.name
            if result_path.exists():
                lines = result_path.read_text().split('\n')
                if len(lines) == 1001:
                    continue

            logging.info(f'Now running {f_path}')
            cmd = ['make', 'ccg2lambda', f'file={f_path.name}']
            # cmd = ['make', 'ccg2lambda', f'file={f_path.name}', f'gpu={len(process_list)}']
            process_list.append(subprocess.Popen(cmd, stdout=subprocess.PIPE))

            # Wait until all process finished
            while len(process_list) == job_num:
                for p in process_list:
                    if p.poll() is not None:
                        process_list.remove(p)
                time.sleep(10)
    except KeyboardInterrupt:
        print('Now killing process... wait')
        for p in process_list:
            p.send_signal(signal.SIGINT)
    return None


if __name__ == "__main__":
    main()
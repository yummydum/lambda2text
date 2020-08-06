import argparse
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


def main():

    args = set_args()

    source_dir = DATA_DIR / f'{args.data}_split'
    result_dir = DATA_DIR / f'{args.data}_formal_split'
    if not result_dir.exists():
        result_dir.mkdir()

    process_list = []
    try:
        for f_path in tqdm(sorted(source_dir.iterdir())):

            t_path = result_dir / f_path.name

            # May skip if the result is already present
            result_path = result_dir / f_path.name
            if result_path.exists():
                lines = result_path.read_text().split('\n')
                if len(lines) == 401:
                    logging.info(
                        f'Skip {result_path} since result already exists')
                    continue
                else:
                    logging.info(
                        f'{result_path} exists but the length is {len(lines)}')
                    pass

            breakpoint()

            logging.info(f'Now running {f_path}')
            cmd = ['make', 'ccg2lambda', f'src={f_path}', f'trg={t_path}']
            if args.gpu:
                cmd += [f'gpu={len(process_list)}']
            process_list.append(subprocess.Popen(cmd, stdout=subprocess.PIPE))

            # Wait until all process finished
            while len(process_list) == args.job_num:
                for p in process_list:
                    if p.poll() is not None:
                        process_list.remove(p)
                time.sleep(10)
    except KeyboardInterrupt:
        print('Now killing process... wait')
        for p in process_list:
            p.send_signal(signal.SIGINT)
    return None


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('job_num')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

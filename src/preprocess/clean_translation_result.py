from config import DATA_DIR

def clean(line):
    return ' '.join(line.replace(' ','').split('▁')).replace('\n','').replace('<eos>','')


def main():
    cleansed_path = DATA_DIR / 'translation_cleansed'/'result.txt' 
    result_path = DATA_DIR / 'translation_log_False_9.txt'
    count = 0
    with cleansed_path.open(mode='w',encoding='utf-8') as f_w:
        with result_path.open(mode='r',encoding='utf-8') as f_r:
            for i, line in enumerate(f_r):
                if i%4 == 2:
                    f_w.write(clean(line) + '\n')
                    count += 1
                
                if count > 1000:
                    break

if __name__ == '__main__':
    main()



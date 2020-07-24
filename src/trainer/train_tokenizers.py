from tokenizers import CharBPETokenizer
from config import DATA_DIR

result_dir = DATA_DIR / 'tokenizers'
if not result_dir.exists():
    result_dir.mkdir()

mrpc_path = DATA_DIR / 'formal/mrpc.txt'
tokenizer = CharBPETokenizer()
tokenizer.train([str(mrpc_path)])
breakpoint()
tokenizer.save(str(result_dir / 'mrpc_tokenizer.json'))
import pandas as pd
from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer
import tokenizers
from config import DATA_DIR

result_dir = DATA_DIR / 'tokenizers'
if not result_dir.exists():
    result_dir.mkdir()
train_data_path = result_dir / 'train_data.txt'

# Create source file
mnli_path = DATA_DIR / 'pairs/mnli.tsv'
df = pd.read_csv(mnli_path, sep='\t')
df['formal'].to_csv(train_data_path, header=False, index=False)

# Train
# tokenizer = BertWordPieceTokenizer()
tokenizer = SentencePieceBPETokenizer()
tokenizer.train([str(train_data_path)])

# Save
tokenizer_dir = result_dir / 'SentencePiece'
if not tokenizer_dir.exists():
    tokenizer_dir.mkdir()
tokenizer.save(str(tokenizer_dir))

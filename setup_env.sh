# Python packages
pip install -r requirements.txt 
pip install torch==1.5.1+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html  # Only linux
# pip install torch==1.5.1 torchvision # for local
pip install -r requirements_torch.txt

# Download models
python -m spacy download en
python -m spacy download de
depccg_en download elmo

# Project package
cd src
pip install -e .
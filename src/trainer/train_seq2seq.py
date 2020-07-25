import argparse
import sys
import random
import logging

import numpy as np
import torch
import wandb
from torch import nn
from config import DATA_DIR

from model.seq2seq import TransformerSeq2Seq
from preprocess.dataset import load_datasets, TEXT, FORMAL
from utils import get_optimzer

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

# Set device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# Set logger
logging.basicConfig(level=logging.DEBUG)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def forward(model, data, criterion):
    """
    Format the data, forward, return the loss
    """
    src = data.formal[0]
    trg = data.text[0]

    # Replace eos to pad to cut off the eos token
    trg_input = trg.clone()
    trg_input[trg_input == 2] = 1
    trg_input = trg_input[:, :-1]  # adjust length

    output, _ = model(src, trg_input)

    # Flatten
    output_dim = output.shape[-1]  # Number of target tokens
    output = output.contiguous().view(-1, output_dim)
    trg_output = trg[:, 1:].contiguous().view(-1)

    #output = [batch size * (trg len - 1), output dim]
    #trg = [batch size * (trg len - 1)]

    return criterion(output, trg_output)


def train(args, model, dataset):
    """
    One epoch for train 
    """

    model.train()
    epoch_loss = 0
    for data in dataset:
        model.zero_grad()

        loss = forward(model, data, args.criterion)

        # Report the loss
        if not args.test_run:
            wandb.log({"train_loss": loss})

        # Update param
        loss.backward()
        args.optimizer.step()

        epoch_loss += loss.item()

        # Just try one loop for test mode
        if args.test_run:
            print("Test succeed for train loop, return.")
            return epoch_loss / 1

    return epoch_loss / len(dataset)


def evaluate(args, model, dataset):
    """
    One epoch for evaluation
    """

    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data in dataset:

            loss = forward(model, data, args.criterion)

            epoch_loss += loss.item()

            # Just try one loop for test mode
            if args.test_run:
                print("Test succeed for eval loop, return.")
                return epoch_loss / 1

    # Report the loss
    if not args.test_run:
        wandb.log({"eval_loss": loss})

    return epoch_loss / len(dataset)


def init_model(args):
    if args.test_run:
        # Small model for fast test
        model = TransformerSeq2Seq(input_dim=len(FORMAL.vocab),
                                   output_dim=len(TEXT.vocab),
                                   hid_dim=14,
                                   n_heads=7,
                                   n_layers=2,
                                   dropout=0.1,
                                   device=DEVICE.type)
    else:
        model = TransformerSeq2Seq(input_dim=len(FORMAL.vocab),
                                   output_dim=len(TEXT.vocab),
                                   hid_dim=args.hid_dim,
                                   n_heads=args.n_heads,
                                   n_layers=args.n_layers,
                                   dropout=args.dropout,
                                   device=DEVICE.type)

    # Handle GPU
    gpu_num = torch.cuda.device_count()
    logging.info(f'Available gpu num: {gpu_num}')
    if torch.cuda.is_available() and gpu_num > 1 and not args.test_run:
        logging.info(f'Use {gpu_num} GPU')
        model = torch.nn.DataParallel(model)
    model = model.to(DEVICE)

    # Init weight
    model.apply(initialize_weights)

    return model


def main():
    args = set_args()

    logging.info('Now loading datasets...')
    train_data, dev_data, test_data = load_datasets(args.batch_size, DEVICE, test_mode=args.test_run)

    logging.info(f'Now initializing model with args {args}')
    model = init_model(args)

    # set up wandb in production mode
    if not args.test_run:
        wandb.init(save_code=True)
        wandb.run.name = wandb.run.id
        wandb.config.update(args)
        wandb.watch(model)

    # criteterion (hard code to cross entropy loss for now)
    args.criterion = nn.CrossEntropyLoss(ignore_index=1)

    # optimizers
    args.optimizer = get_optimzer(model, args.lr, args.decay)

    # Loop over epochs
    logging.info('Start training!')
    for epoch in range(args.epoch_num):
        print(f'Now in {epoch}th epoch')
        train_loss = train(args, model, train_data)
        valid_loss = evaluate(args, model, dev_data)
        if args.test_run:
            print("Test succeed for main, exit.")
            return 0

    logging.info('Finish train, save model...')
    result_path = DATA_DIR / 'trained_model' / f'{wandb.run.name}.pt'
    torch.save(model, str(result_path))
    return None


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--test_run', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
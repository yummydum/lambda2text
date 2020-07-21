import argparse
import sys
import random

import numpy as np
from tqdm import tqdm
import torch
import wandb
from torch import nn

from model.seq2seq import TransformerSeq2Seq
from preprocess.dataset import load_datasets, TEXT, FORMAL
from trainer.utils import get_optimzer

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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

    output, _ = model(src, trg_input)

    # Flatten
    output_dim = output.shape[-1]  # Number of target tokens
    output = output.contiguous().view(-1, output_dim)
    trg_output = trg.contiguous().view(-1)

    #output = [batch size * (trg len - 1), output dim]
    #trg = [batch size * (trg len - 1)]

    return criterion(output, trg_output)


def train(args, model, dataset):
    """
    One epoch for train 
    """

    model.train()
    epoch_loss = 0
    for data in tqdm(dataset):
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
        for data in tqdm(dataset):

            loss = forward(model, data, args.criterion)

            # Report the loss
            if not args.test_run:
                wandb.log({"eval_loss": loss})

            epoch_loss += loss.item()

            # Just try one loop for test mode
            if args.test_run:
                print("Test succeed for eval loop, return.")
                return epoch_loss / 1

    return epoch_loss / len(dataset)


def main():
    args = set_args()

    # Load datasets
    train_data, dev_data, test_data = load_datasets(args.batch_size, device)

    # Init model
    if args.test_run:
        # Test model
        model = TransformerSeq2Seq(input_dim=len(FORMAL.vocab),
                                   output_dim=len(TEXT.vocab),
                                   hid_dim=14,
                                   n_heads=7,
                                   n_layers=2)
    else:
        raise NotImplementedError()
        model = ''

    # Init weight
    model.apply(initialize_weights)

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
    for epoch in range(args.epoch_num):
        print(f'Now in {epoch}th epoch')
        train_loss = train(args, model, train_data)
        valid_loss = evaluate(args, model, dev_data)
        if args.test_run:
            print("Test succeed for main, exit.")
            sys.exit(0)
    return None


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dim', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--test_run', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
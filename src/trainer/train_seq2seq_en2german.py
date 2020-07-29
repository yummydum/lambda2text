import argparse
import random
import logging

import numpy as np
import torch
from torch import nn
import wandb
from pytorch_memlab import LineProfiler

from config import DATA_DIR
from model.seq2seq import TransformerSeq2Seq
from preprocess.dataset_multi30k import load_Multi30k, ENGLISH,GERMAN
from utils import get_optimzer,translation_example

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

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

    src = data.src
    trg = data.trg

    # Replace eos to pad to cut off the eos token
    trg_input = trg.clone()
    trg_input[trg_input == 2] = 1
    trg_input = trg_input[:, :-1]  # adjust length

    # Forward
    output, _ = model(src, trg_input)

    # Flatten output
    output_dim = output.shape[-1]  # Number of target tokens
    output = output.contiguous().view(-1, output_dim)

    # Make golden output
    trg_output = trg[:, 1:].contiguous().view(-1).detach()

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
            wandb.log({"train_loss": loss.item()})

        # Update param
        loss.backward()
        args.optimizer.step()

        epoch_loss += loss.item()

        # Just try one loop for test mode
        if args.test_run:
            logging.info("Test succeed for train loop, return.")
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
                logging.info("Test succeed for eval loop, return.")
                return epoch_loss / 1

    result = epoch_loss / len(dataset)
    # Report the loss
    if not args.test_run:
        wandb.log({"eval_loss": result})

    return result


def init_model(args):
    if args.test_run:
        # Small model for fast test
        model = TransformerSeq2Seq(input_dim=len(GERMAN.vocab),
                                   output_dim=len(ENGLISH.vocab),
                                   hid_dim=14,
                                   n_heads=7,
                                   n_layers=2,
                                   dropout=0.1,
                                   device=DEVICE.type)
    else:
        model = TransformerSeq2Seq(input_dim=len(GERMAN.vocab),
                                   output_dim=len(ENGLISH.vocab),
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
    train_data, dev_data, test_data = load_Multi30k(args.batch_size,
                                                    DEVICE)

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
    #with LineProfiler(train, forward) as prof:
    for epoch in range(args.epoch_num):
        logging.info(f'Now in {epoch}th epoch')

        # Train & eval
        train(args, model, train_data)
        evaluate(args, model, dev_data)

        # Log gpu usage
        # logging.info(prof.display())

        # Log example translation
        # translation_example(test_data,model,TEXT,FORMAL,DEVICE)

        if args.test_run:
            break

    test_loss = evaluate(args,model,test_data)

    # End of process
    if args.test_run:
        logging.info('Test run succeed, exit')
        return 0
    else:
        if not args.test_run:
            wandb.log({"test_loss": test_loss})
        logging.info('Saving model...')
        result_path = DATA_DIR / 'trained_model' / f'{wandb.run.name}.pt'
        torch.save(model, str(result_path))
    return 0


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--n_layers', default=6, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=30, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--test_run', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()

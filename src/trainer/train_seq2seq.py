import argparse
import random
import logging

import numpy as np
import torch
from torch import nn
import wandb

from config import DATA_DIR
from model import transformer_seq2seq, lstm_seq2seq, gru_seq2seq
from utils import calculate_bleu
from preprocess.dataset import load_data

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
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

    if isinstance(model,torch.nn.DataParallel):
        name = model.module.name
    else:
        name = model.name

    if name == 'transformer':

        src = data.src[0]
        trg = data.trg[0]

        # Forward
        output, _ = model(src, trg[:, :-1])

        # Flatten output
        output_dim = output.shape[-1]  # Number of target tokens
        output = output.contiguous().view(-1, output_dim)

        # Make golden output
        trg = trg[:, 1:].contiguous().view(-1)
        return criterion(output, trg)
<<<<<<< HEAD
    
    elif name in {'lstm','gru'}:
        src,src_len = data.src
=======

    elif model.name == 'lstm':
        src, src_len = data.src
>>>>>>> 8827df35f25b62e0be55d735a2ca43f6d788ce6b
        trg = data.trg[0]

        # Forward
        src = src.transpose(0,1)
        trg = trg.transpose(0,1)
        output = model(src, src_len, trg)

        # Reshape
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)

        # Make golden output
<<<<<<< HEAD
        trg = trg[1:].contiguous().view(-1)
    
=======
        trg = trg.transpose(0, 1)[1:].contiguous().view(-1)

>>>>>>> 8827df35f25b62e0be55d735a2ca43f6d788ce6b
        return criterion(output, trg)
    
    else:
        raise ValueErorr('Choose transformer or lstm')


def train(args, model, dataset):
    """
    One epoch for train 
    """

    model.train()
    epoch_loss = 0
    for data in dataset:

        args.optimizer.zero_grad()

        loss = forward(model, data, args.criterion)

        # Report the loss
        if not args.test_run:
            wandb.log({"train_loss": loss.item()})

        # Update param
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
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


def init_model(args, src_field, trg_field):
    input_dim = len(src_field.vocab)
    output_dim = len(trg_field.vocab)
    if args.test_run:
        # Small model for fast test

        if args.model == 'transformer':
            model = transformer_seq2seq.Seq2Seq(input_dim=input_dim,
                                                output_dim=output_dim,
                                                hid_dim=14,
                                                n_heads=7,
                                                n_layers=2,
                                                dropout=0.1,
                                                device=DEVICE,
                                                pf_dim=512,
                                                max_len=1000)

        elif args.model == 'lstm':
            model = lstm_seq2seq.Seq2Seq(input_dim=input_dim,
                                         output_dim=output_dim,
                                         hid_dim=14,
                                         dropout=0.1,
                                         device=DEVICE)

        elif args.model == 'gru':
            model = gru_seq2seq.Seq2Seq(input_dim=input_dim,
            output_dim=output_dim,
            hid_dim = 14,dropout=0.1,device=DEVICE)
        

    else:
        if args.model == 'transformer':
            model = transformer_seq2seq.Seq2Seq(input_dim=len(src_field.vocab),
                                                output_dim=len(
                                                    trg_field.vocab),
                                                hid_dim=args.hid_dim,
                                                n_heads=args.n_heads,
                                                n_layers=args.n_layers,
                                                dropout=args.dropout,
                                                device=DEVICE,
                                                pf_dim=512,
                                                max_len=1000)
        elif args.model == 'lstm':
            model = lstm_seq2seq.Seq2Seq(input_dim=input_dim,
                                         output_dim=output_dim,
                                         hid_dim=args.hid_dim,
                                         dropout=0.1,
                                         device=DEVICE)

        elif args.model == 'gru':
            model = gru_seq2seq.Seq2Seq(input_dim=input_dim,
            output_dim=output_dim,
            hid_dim = args.hid_dim,dropout=0.1,device=DEVICE)

    # Handle GPU
    gpu_num = torch.cuda.device_count()
    logging.info(f'Available gpu num: {gpu_num}')
    if torch.cuda.is_available() and gpu_num > 1 and not args.test_run:
        logging.info(f'Use {gpu_num} GPU')
        model = torch.nn.DataParallel(model)
    else:
        logging.info(f'Use single GPU')
    model = model.to(DEVICE)

    # Init weight
    model.apply(initialize_weights)

    return model


def main():
    args = set_args()

    logging.info('Now loading datasets...')
    data, SRC, TRG = load_data(
        args.data,
        args.batch_size,
        DEVICE,
        test_mode=args.test_run,
    )
    train_data, dev_data, test_data = data

    logging.info(f'Now initializing model with args {args}')
    model = init_model(args, SRC, TRG)

    # set up wandb in production mode
    if not args.test_run:
        wandb.init(save_code=True)
        wandb.run.name = wandb.run.id
        wandb.config.update(args)
        wandb.watch(model)

    # criteterion (hard code to cross entropy loss for now)
    args.criterion = nn.CrossEntropyLoss(ignore_index=1)

    # optimizers
    args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Loop over epochs
    logging.info('Start training!')
    for epoch in range(args.epoch_num):
        logging.info(f'Now in {epoch}th epoch')
        epoch_trans_path = DATA_DIR / 'translation' / f'translation_log_{args.data}_{args.model}.txt'

        # Train & eval
        train(args, model, train_data)
        evaluate(args, model, dev_data)

        if args.test_run:
            # Only one epoch for test run
            break
        else:
            result_path = DATA_DIR / 'trained_model' / f'{wandb.run.name}_{epoch}.pt'
            logging.info(f'Saving model to {result_path}')
            torch.save(model, str(result_path))

    logging.info('Finish process')
    test_loss = evaluate(args, model, test_data)
    bleu = calculate_bleu(dev_data.dataset,
<<<<<<< HEAD
                        SRC,
                        TRG,
                        model,
                        DEVICE,
                        trans_path=epoch_trans_path,
                        formula=args.formula)
=======
                          SRC,
                          TRG,
                          model,
                          DEVICE,
                          trans_path=epoch_trans_path,
                          formula=args.formula,
                          limit=1000)
>>>>>>> 8827df35f25b62e0be55d735a2ca43f6d788ce6b

    if not args.test_run:
        wandb.log({"test_loss": test_loss})
        wandb.log({"bleu": bleu * 100})
    return 0


def set_args():
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('data', choices=['snli','mnli','m30k'])
    parser.add_argument('model', choices=['transformer','lstm','gru'])
=======
    parser.add_argument('data',
                        choices=[
                            'snli', 'mnli', 'm30k', '2018_simple',
                            '2018_formula', '2018_graph'
                        ])
    parser.add_argument('model', choices=['transformer', 'lstm'])
>>>>>>> 8827df35f25b62e0be55d735a2ca43f6d788ce6b
    parser.add_argument('--hid_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--n_layers', default=3, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--test_run', action='store_true')
    args = parser.parse_args()
<<<<<<< HEAD
    args.formula = args.data in {'snli','mnli'}
    if args.test_run:
        args.batch_size = 4
=======
    args.formula = args.data in {'snli', 'mnli'}
>>>>>>> 8827df35f25b62e0be55d735a2ca43f6d788ce6b
    return args


if __name__ == '__main__':
    main()

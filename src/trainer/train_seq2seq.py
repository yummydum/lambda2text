import argparse
import random

import numpy as np
from tqdm import tqdm
import torch
from transformers import AlbertConfig
import wandb

from model.seq2seq import AlbertSeq2Seq
from preprocess.dataset import read_pairs
from trainer.utils import get_optimzer, get_scheduler, hinge_loss

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def main():

    args = set_args()
    config = AlbertConfig(vocab_size=17229)

    # Init model
    model = AlbertSeq2Seq(config).to(device)

    # set up wandb
    wandb.init(save_code=True)
    wandb.run.name = wandb.run.id
    wandb.config.update(args)
    wandb.watch(model)

    # Load datasets
    dataset = read_pairs(args.batch_size, device)

    # Loss, optimizer
    num_step = len(dataset) * args.epoch_num
    optimizer = get_optimzer(model, args.lr, args.decay)
    scheduler = get_scheduler(optimizer, num_step, args.warmup_step)

    # Train
    for epoch in range(args.epoch_num):
        print(f'Now in {epoch}th epoch')
        for data in tqdm(dataset):

            model.train()
            model.zero_grad()

            # Calc the loss
            pred = model(data.formal[0])
            loss = hinge_loss(pred, data.encoded)

            # Reprt the loss
            wandb.log({"train_loss": loss})

            # Update param
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval dev loss
        # model.eval()
        # with torch.no_grad():
        #     for data in tqdm():

        #         # Predict
        #         p_pred, a_pred, attention = model(data)
        #         p_pred = p_pred.squeeze(dim=1)

        #         # Calc loss
        #         a_golden = atomic.get(data.pair_id).to(device)
        #         p_loss = utils.hinge_loss(p_pred, data.labels)

        #         # Get pred
        #         p_pred = utils.get_pred(p_pred)

        #         # Calc metrics

        # logged = {'p_acc': p_acc / dev_data_len, 'dev_loss': dev_loss}
        # wandb.log(logged)

    return None


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--warmup_step', default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
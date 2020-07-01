import argparse
from pathlib import Path
import random
import sys
sys.path.append('..')

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AlbertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import wandb

from models.seq2seq import AlbertSeq2seq
from preprocess.dataset import FormalSurfacePair

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

    # set up wandb
    wandb.init(save_code=True)
    wandb.run.name = wandb.run.id
    wandb.config.update(args)

    # Init model
    model = AlbertSeq2seq(config).to(device)
    wandb.watch(model)

    # Load datasets
    dataset = FormalSurfacePair()
    dataset = DataLoader(dataset, shuffle=True, device=device)

    # Loss, optimizer
    num_step = len(dataset) * args.epoch_num
    optimizer = utils.get_optimzer(model, args.lr, args.decay)
    scheduler = utils.get_scheduler(optimizer, num_step, args.warmup_step)

    # Train
    count = 0
    for epoch in range(args.epoch_num):
        print(f'Now in {epoch}th epoch')
        for data in tqdm(dataset):

            model.train()
            model.zero_grad()

            # Calc the loss
            pred = model(data)
            p_loss = utils.hinge_loss(p_pred, data.labels)

            # Reprt the loss
            wandb.log({"p_loss": p_loss, "a_loss": a_loss, "train_loss": loss})

            # Update param
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Eval dev loss
        model.eval()
        with torch.no_grad():
            for data in tqdm(ETPC_dev):

                # Predict
                p_pred, a_pred, attention = model(data)
                p_pred = p_pred.squeeze(dim=1)

                # Calc loss
                a_golden = atomic.get(data.pair_id).to(device)
                p_loss = utils.hinge_loss(p_pred, data.labels)

                # Get pred
                p_pred = utils.get_pred(p_pred)

                # Calc metrics

        logged = {'p_acc': p_acc / dev_data_len, 'dev_loss': dev_loss}
        wandb.log(logged)

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
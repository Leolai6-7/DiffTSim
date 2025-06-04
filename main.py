import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.simsiam import SimSiam
from models.transformer import TransformerEncoder
from models.unet import UNet
from train.train_diffusion import pretrain_diffusion
from train.train_traditional import pretrain_traditional
from utils.dataset import TimeSeriesDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['diffusion', 'traditional'], default='diffusion')
    parser.add_argument('--data', type=str, default='data.npy')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    dataset = TimeSeriesDataset(args.data, args.label)

    # encoder + simsiam
    encoder = TransformerEncoder(input_dim=dataset[0][0].shape[0])
    model = SimSiam(encoder)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.mode == 'diffusion':
        diffusion_model = UNet(in_channels=dataset[0][0].shape[0])
        pretrain_diffusion(model, diffusion_model, dataset, optimizer, device, epochs=args.epochs, batch_size=args.batch_size)
    else:
        pretrain_traditional(model, dataset, optimizer, device, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == '__main__':
    main()

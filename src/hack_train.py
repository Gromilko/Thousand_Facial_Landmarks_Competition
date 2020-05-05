"""Script for baseline training. Model is ResNet18 (pretrained on ImageNet). Training takes ~ 15 mins (@ GTX 1080Ti)."""

import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import tqdm
from torch.nn import functional as fnn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.EarlyStopping import EarlyStopping
from src.hack_utils import NUM_PTS, CROP_SIZE
from src.hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys, HorizontalFlip
from src.hack_utils import ThousandLandmarksDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

writer = SummaryWriter('runs/trainer_{}'.format(time.strftime("%y-%m-%d %H:%M", time.gmtime())))


# resnet50 bs = 350

def parse_arguments():
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name (for saving checkpoints and submits).",
                        default="baseline")
    parser.add_argument("--data", "-d", help="Path to dir with target images & landmarks.", default=None)
    parser.add_argument("--batch-size", "-b", default=512, type=int)  # 512 is OK for resnet18 finetune @ 6Gb of VRAM
    parser.add_argument("--epochs", "-e", default=1, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


def train(model, loader, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="training..."):
        images = batch["image"].to(device)  # B x 3 x CROP_SIZE x CROP_SIZE
        landmarks = batch["landmarks"]  # B x (2 * NUM_PTS)

        pred_landmarks = model(images).cpu()  # B x (2 * NUM_PTS)
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(train_loss)


def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = []
    for batch in tqdm.tqdm(loader, total=len(loader), desc="validation..."):
        images = batch["image"].to(device)
        landmarks = batch["landmarks"]

        with torch.no_grad():
            pred_landmarks = model(images).cpu()
        loss = loss_fn(pred_landmarks, landmarks, reduction="mean")
        val_loss.append(loss.item())

    return np.mean(val_loss)


class Trainer:
    def __init__(self, model, dataloaders):
        self.device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
        self.model = model.to(self.device)

        self.dataloaders = dataloaders

        self.optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                                    lr=args.learning_rate, amsgrad=True)
        self.loss_fn = fnn.mse_loss

        self.hist_dir = os.path.join('history', 'weights', args.name)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        # self.scheduler = CyclicLR(self.optimizer, )

        self.early_stopping = EarlyStopping(patience=4, verbose=True)

        for phase in ["train", "val"]:
            print(f'{phase} dataset len=', len(self.dataloaders[phase]))

    def start(self, start_with=0):
        # 2. train & validate
        print("Ready for training...")
        best_val_loss = np.inf
        for epoch in range(start_with, args.epochs + start_with):
            train_loss = train(self.model, self.dataloaders['train'], self.loss_fn, self.optimizer, device=self.device)
            val_loss = validate(self.model, self.dataloaders['val'], self.loss_fn, device=self.device)

            writer.add_scalars('loss', {'train': train_loss,
                                        'val': val_loss}, epoch)

            print("Epoch #{:2}:\ttrain loss: {:7.4}\tval loss: {:7.4}".format(epoch, train_loss, val_loss))

            self.scheduler.step(val_loss, epoch)
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            for group_num, param_group in enumerate(self.optimizer.param_groups):
                writer.add_scalar('lr/{}'.format(group_num), param_group['lr'], epoch)

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(self.hist_dir):
                    os.mkdir(self.hist_dir)
                with open(os.path.join(self.hist_dir, "ep{}_loss{:.4}.pth".format(epoch, val_loss)), "wb") as fp:
                    torch.save(self.model.state_dict(), fp)

        return epoch+1

train_transforms = transforms.Compose([
    ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
    CropCenter(CROP_SIZE),
    HorizontalFlip(0.5),
    TransformByKeys(transforms.ToPILImage(), ("image",)),
    TransformByKeys(transforms.ToTensor(), ("image",)),
    TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
])


def data_provider(split=None):
    print(f"Reading {split} data...")
    if split == "train":
        train_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'),
                                                 train_transforms, split="train")
        dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                     pin_memory=True, shuffle=True, drop_last=True)
    elif split == "val":
        val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'),
                                               train_transforms, split="val")
        dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                                     pin_memory=True, shuffle=False, drop_last=False)
    else:
        # TODO: exception
        print('Bad split value')

    return dataloader


def get_stat(model):
    print()
    count = 0
    for param in model.parameters():
        count += 1
    print('Вcего параметров', count)

    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += 1
    print('True параметров', count)

    count = 0
    for param in model.parameters():
        if not param.requires_grad:
            count += 1
    print('False параметров', count)
    print()


def freeze_layers(model):
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # print('train only head')
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    # trainer = Trainer(model, dataloaders)
    # get_stat(model)
    # args.epochs = 5
    # end_epoch = trainer.start()

    # print('train layer4 and head')
    # args.epochs = 10
    # train only head
    for param in model.layer4.parameters():
        param.requires_grad = True

    # trainer = Trainer(model, dataloaders)
    # get_stat(model)
    # end_epoch = trainer.start(start_with=end_epoch)
    #
    # print('train layer3 and head')
    # train only head
    for param in model.layer3.parameters():
        param.requires_grad = True

    # args.epochs = 15
    # trainer = Trainer(model, dataloaders)
    # get_stat(model)
    # end_epoch = trainer.start(start_with=end_epoch)

    print('train layer2 and head')
    # train only head
    for param in model.layer2.parameters():
        param.requires_grad = True

    # with open(f"history/weights/resnet50_layer_wise/ep38_loss1.745.pth", "rb") as fp:
    #     best_state_dict = torch.load(fp, map_location="cpu")
    #     model.load_state_dict(best_state_dict)


def main(args):
    args.batch_size = 4
    dataloader = data_provider("train")

    print("Creating model...")
    model = models.resnext101_32x8d(pretrained=True, )
    model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)

    name = 'history/weights/resneXt101_234layer/ep18_loss1.54'
    with open(f"{name}.pth", "rb") as fp:
        best_state_dict = torch.load(fp, map_location="cpu")
        model.load_state_dict(best_state_dict)

    dataloaders = {phase: data_provider(phase) for phase in ["train", "val"]}

    args.epochs = 20
    trainer = Trainer(model, dataloaders)
    get_stat(model)
    end_epoch = trainer.start()

    input('vse')


if __name__ == '__main__':
    print(os.getcwd())
    args = parse_arguments()
    sys.exit(main(args))

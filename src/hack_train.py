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
from torchvision.models._utils import IntermediateLayerGetter


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
    def __init__(self, model, fold):
        self.device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
        self.model = model.to(self.device)

        self.fold = fold

        self.dataloaders = {phase: data_provider(phase, fold) for phase in ["train", "val"]}

        self.optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                                    lr=args.learning_rate, amsgrad=True)
        self.loss_fn = fnn.mse_loss

        self.hist_dir = os.path.join('history', 'weights', args.name)

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        # self.scheduler = CyclicLR(self.optimizer, )

        self.early_stopping = EarlyStopping(patience=5, verbose=True)

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

            # self.scheduler.step(val_loss, epoch)
            # self.early_stopping(val_loss, self.model)
            #
            # if self.early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            for group_num, param_group in enumerate(self.optimizer.param_groups):
                writer.add_scalar('lr/{}'.format(group_num), param_group['lr'], epoch)

            # if val_loss <= best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(self.hist_dir):
                os.mkdir(self.hist_dir)
            with open(os.path.join(self.hist_dir, "fold{}_ep{}_loss{:.4}.pth".format(self.fold, epoch, val_loss)), "wb") as fp:
                torch.save(self.model.state_dict(), fp)

        with open(os.path.join(self.hist_dir, "finish.pth"), "wb") as fp:
            torch.save(self.model.state_dict(), fp)

        return epoch + 1


train_transforms = transforms.Compose([
    ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
    CropCenter(CROP_SIZE),
    # HorizontalFlip(0.5),
    TransformByKeys(transforms.ToPILImage(), ("image",)),
    TransformByKeys(transforms.ToTensor(), ("image",)),
    TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
])


def data_provider(split=None, fold=None):
    print(f"Reading {split} data for fold={fold}...")
    if split == "train":
        train_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'),
                                                 train_transforms, split="train", fold=fold)
        dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                     pin_memory=True, shuffle=True, drop_last=True)
    elif split == "val":
        val_dataset = ThousandLandmarksDataset(os.path.join(args.data, 'train'),
                                               train_transforms, split="val", fold=fold)
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
    # model = torchvision.models.(pretrained=True)
    # # https://discuss.pytorch.org/t/how-to-perform-finetuning-in-pytorch/419/19
    # features = nn.Sequential(*list(model.children())[:-2])
    # out = features(torch.rand(1, 3, 224, 224))
    # # extract layer1 and layer3, giving as names `feat1` and feat2`
    # return_layers = {'conv1': 'conv1', 'maxpool': 'maxpool',   'layer1': '1', 'layer2': '2', 'layer3': '3', 'layer4': '4'}
    # new_m = torchvision.models._utils.IntermediateLayerGetter(model, return_layers=return_layers)
    # out = new_m(torch.rand(1, 3, 224, 224))
    # print([(k, v.shape) for k, v in out.items()])

    for fold in range(0, 1):
        print("Creating model...")
        model = models.resnet101(pretrained=True, )
        model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)

        # models.detection.keypointrcnn_resnet50_fpn()

        # name = 'history/weights/resneXt101_234layer/ep18_loss1.54'
        # with open(f"{name}.pth", "rb") as fp:
        #     best_state_dict = torch.load(fp, map_location="cpu")
        #     model.load_state_dict(best_state_dict)

        args.epochs = 23
        args.batch_size = 240
        trainer = Trainer(model, fold=fold)
        get_stat(model)
        end_epoch = trainer.start()

    input('vse')


if __name__ == '__main__':
    print(os.getcwd())
    args = parse_arguments()
    sys.exit(main(args))

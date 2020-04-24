import os

import torch
from torch import nn, optim
from torch.nn import functional as fnn
from torch.utils import data
from torchvision import models, transforms

from hack_train import predict
from hack_utils import ThousandLandmarksDataset

from hack_utils import NUM_PTS, CROP_SIZE
from hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from hack_utils import ThousandLandmarksDataset
from hack_utils import restore_landmarks_batch, create_submission

train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
    ])

print("Creating model...")
device = torch.device("cuda: 0")
model = models.resnet152(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
model.to(device)


# 3. predict
test_dataset = ThousandLandmarksDataset(os.path.join('data', 'test'), train_transforms, split="test")
test_dataloader = data.DataLoader(test_dataset, batch_size=160, num_workers=0, pin_memory=True,
                                  shuffle=False, drop_last=False)

name = 'resnet152_pretrain3ep_plus_6ep_8ep_bs160_ep4_loss1.508120443105214_best'
with open(f"{name}.pth", "rb") as fp:
    best_state_dict = torch.load(fp, map_location="cpu")
    model.load_state_dict(best_state_dict)

test_predictions = predict(model, test_dataloader, device)

create_submission('data', test_predictions, f"{name}_submit.csv")

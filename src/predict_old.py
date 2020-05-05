import os
import pickle

import torch
from torch import nn
from torch.utils import data
from torchvision import models, transforms

from src.hack_train import predict

from src.hack_utils import NUM_PTS, CROP_SIZE
from src.hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from src.hack_utils import ThousandLandmarksDataset
from src.hack_utils import create_submission

train_transforms = transforms.Compose([
        ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
        CropCenter(CROP_SIZE),
        TransformByKeys(transforms.ToPILImage(), ("image",)),
        TransformByKeys(transforms.ToTensor(), ("image",)),
        TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
    ])

print("Creating model...")
device = torch.device("cuda: 0")
model = models.resnext101_32x8d(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
model.to(device)


# 3. predict
test_dataset = ThousandLandmarksDataset(os.path.join('../data', 'test'), train_transforms, split="test")
test_dataloader = data.DataLoader(test_dataset, batch_size=256, num_workers=0, pin_memory=True,
                                  shuffle=False, drop_last=False)

name = '../history/weights/resneXt101_pretrain_start_18/ep6_loss1.503'
with open(f"{name}.pth", "rb") as fp:
    best_state_dict = torch.load(fp, map_location="cpu")
    model.load_state_dict(best_state_dict)

test_predictions = predict(model, test_dataloader, device)

with open(f"{name}_test_predictions.pkl", "wb") as fp:
    pickle.dump({"image_names": test_dataset.image_names,
                 "landmarks": test_predictions}, fp)

create_submission('../data', test_predictions, f"{name}_submit.csv")

import os
import pickle

import numpy as np
import torch
import tqdm
from torch import nn
from torch.utils import data
from torchvision import models, transforms

from src.hack_utils import NUM_PTS, CROP_SIZE, restore_landmarks_batch
from src.hack_utils import ScaleMinSideToSize, CropCenter, TransformByKeys
from src.hack_utils import ThousandLandmarksDataset
from src.hack_utils import create_submission

train_transforms = transforms.Compose([
    ScaleMinSideToSize((CROP_SIZE, CROP_SIZE)),
    CropCenter(CROP_SIZE),
    TransformByKeys(transforms.ToPILImage(), ("image",)),
    TransformByKeys(transforms.ToTensor(), ("image",)),
    # TransformByKeys(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ("image",)),
])


class Predictor:
    def __init__(self, model, loader, device):
        self.model = model
        self.loader = loader
        self.device = device
        self.aug = None

    def __call__(self, tta=False):
        self.model.eval()
        if tta:
            raise NotImplementedError
        else:
            predictions = np.zeros((len(self.loader.dataset), NUM_PTS, 2))
            for i, batch in enumerate(tqdm.tqdm(self.loader, total=len(self.loader), desc="test prediction...")):
                images = batch["image"].to(self.device)

                with torch.no_grad():
                    pred_landmarks = model(images).cpu()
                pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))  # B x NUM_PTS x 2

                fs = batch["scale_coef"].numpy()  # B
                margins_x = batch["crop_margin_x"].numpy()  # B
                margins_y = batch["crop_margin_y"].numpy()  # B
                prediction = restore_landmarks_batch(pred_landmarks, fs, margins_x, margins_y)  # B x NUM_PTS x 2
                predictions[i * self.loader.batch_size: (i + 1) * self.loader.batch_size] = prediction
        return predictions


print("Creating model...")
device = torch.device("cuda: 0")
torch.cuda.set_device(device)

model = models.resnet101(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2 * NUM_PTS, bias=True)
model.to(device)

# 3. predict
test_dataset = ThousandLandmarksDataset(os.path.join('../data', 'test'), train_transforms, split="test")
test_dataloader = data.DataLoader(test_dataset, batch_size=1024, num_workers=0, pin_memory=True,
                                  shuffle=False, drop_last=False)

name = '../history/weights/finetuning_albu/fold0_ep6_loss1.14'
with open(f"{name}.pth", "rb") as fp:
    best_state_dict = torch.load(fp, map_location="cpu")
    model.load_state_dict(best_state_dict)

predictor = Predictor(model, test_dataloader, device)
test_predictions = predictor(tta=False)

# with open(f"{name}_test_predictions.pkl", "wb") as fp:
#     pickle.dump({"image_names": test_dataset.image_names,
#                  "landmarks": test_predictions}, fp)

create_submission('../data', test_predictions, f"{name}_submit.csv")

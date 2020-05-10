import os

import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils import data

from sklearn.model_selection import KFold

import albumentations as albu

np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SIZE = 0.95
NUM_PTS = 971
CROP_SIZE = 128
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"

LEN_DF = 393930
CHUNK_SIZE = 50000


class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
        self.size = torch.tensor(size, dtype=torch.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class HorizontalFlip(object):
    def __init__(self, p=0.5, elem_name='image'):
        self.p = p
        self.elem_name = elem_name

    def __call__(self, sample):
        # pred_landmarks = pred_landmarks.numpy().reshape((len(pred_landmarks), NUM_PTS, 2))
        augmented = albu.Compose([albu.HorizontalFlip(p=self.p)],
                                 keypoint_params=albu.KeypointParams(format='xy')
                                 )(force_apply=True,
                                   image=sample[self.elem_name],
                                   keypoints=torch.reshape(sample['landmarks'], (NUM_PTS, 2)))

        sample[self.elem_name], sample['landmarks'] = augmented['image'], torch.reshape(augmented['keypoints'], (NUM_PTS*2))
        return sample


class MyCoarseDropout(object):
    def __init__(self, p=0.5, elem_name='image'):
        self.p = p
        self.elem_name = elem_name

    def __call__(self, sample):
        augmented = albu.Compose([albu.CoarseDropout(max_holes=4, max_height=20, max_width=20,
                                                     min_holes=2, min_height=10, min_width=10,
                                                     fill_value=np.random.randint(0, 255, 1), p=self.p)],
                                 )(image=sample[self.elem_name],)

        sample[self.elem_name] = augmented['image']
        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class ThousandLandmarksDataset(data.Dataset):

    def __init__(self, root, transforms, split="train", fold=None):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split is not "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for line in fp)

        print(f"N_rows: {num_lines}")

        # train_set = set()
        # val_set = set()

        # kf = KFold(n_splits=5, random_state=42, shuffle=True)
        # for i, (train_index, val_index) in enumerate(kf.split(range(num_lines))):
        #     if i == fold:
        #         train_set.update(set(train_index))
        #         val_set.update(set(val_index))
        #         print(f"Spliting data for fold {fold}. TRAIN: {len(train_index)}. TEST: {len(val_index)}")
        #         break

        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp)):
                if i > 256:
                    break
                if i == 0:
                    continue  # skip header
                if split == "train" and i == int(TRAIN_SIZE * num_lines):
                    break  # reached end of train part of data
                elif split == "val" and i < int(TRAIN_SIZE * num_lines):
                    continue  # has not reached start of val part of data
                # if i > 1024:
                #     break
                # if i == 0:
                #     continue  # skip header
                # if split == "train" and i not in train_set:  # == int(TRAIN_SIZE * num_lines):
                #     continue  # reached end of train part of data
                # elif split == "val" and i not in val_set:  # < int(TRAIN_SIZE * num_lines):
                #     continue  # has not reached start of val part of data
                # sdfsdfываыва
                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                self.image_names.append(image_name)

                if split in ("train", "val"):
                    landmarks = list(map(np.int16, elements[1:]))
                    landmarks = np.array(landmarks, dtype=np.int16).reshape((len(landmarks) // 2, 2))
                    self.landmarks.append(landmarks)

        print('Convert to tensor...', end=' ')
        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms
        print(f'finish')

    # def __init__(self, root, transforms, split="train", **kwargs):
    #     super(ThousandLandmarksDataset, self).__init__()
    #     self.root = root
    #     landmark_file_name = os.path.join(root, 'landmarks.csv') if split is not "test" \
    #         else os.path.join(root, "test_points.csv")
    #     images_root = os.path.join(root, "images")
    #
    #     # 393930 the number of rows in the training dataset
    #     # change to a smaller number if you need to learn from a piece of data
    #     n_rows = 2048 if split is not "test" else None
    #     print(f"Cook {split} data from csv...")
    #
    #     df_chunk = pd.read_csv(landmark_file_name, nrows=n_rows, chunksize=CHUNK_SIZE, delimiter='\t', )
    #     self.landmarks = []
    #     self.image_names = []
    #
    #     print(f"Chunk...", end=' ')
    #     for i, chunk in enumerate(df_chunk):
    #         split_idxs = {"train": range(0, int(TRAIN_SIZE * len(chunk))),
    #                       "val": range(int(TRAIN_SIZE * len(chunk)), len(chunk)),
    #                       "test": range(len(chunk))}
    #         idxs = split_idxs[split]
    #
    #         print(f'{i}...', end=' ')
    #         if split in ("train", "val"):
    #             for row in chunk._values[idxs]:
    #                 self.image_names.append(os.path.join(images_root, row[0]))
    #                 self.landmarks.append(row[1:].astype('int32').reshape((len(row) // 2, 2)))
    #         elif split == 'test':
    #             for row in chunk._values[idxs]:
    #                 self.image_names.append(os.path.join(images_root, row[0]))
    #             self.landmarks = None
    #         else:
    #             raise NotImplementedError(split)
    #     print(f'finish')
    #
    #     print('Convert to tensor...', end=' ')
    #     if split in ("train", "val"):
    #         self.landmarks = torch.as_tensor(self.landmarks)
    #     elif split == 'test':
    #         self.landmarks = None
    #     else:
    #         raise NotImplementedError(split)
    #
    #     self.transforms = transforms
    #     print('finish')

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')

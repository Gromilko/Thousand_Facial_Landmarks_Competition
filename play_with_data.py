import os

import pandas as pd

root = os.path.join('data', 'train')

landmark_file_name = os.path.join(root, 'landmarks.csv')
images_root = os.path.join(root, "images")


n_row = []

df_chunk = pd.read_csv(landmark_file_name, nrows=None, chunksize=50000, delimiter='\t', )
for chunk in df_chunk:
    print(chunk.shape[0])
    n_row.append(chunk.shape[0])

print(sum(n_row))

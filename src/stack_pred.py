import pandas as pd
from tqdm import tqdm

submit_list = ['../stack/resnet152_pretrain3ep_plus_6ep_bs160_ep8_loss1.5298880638505576_best_submit.csv',  # 9.51835
               '../stack/resnet101_pretrain_bs240_ep14_loss1.527_submit.csv',  # 9.32149
               # '../stack/resnet50_pretrain_bs350_submit.csv',  # 9.61028
               # '../history/weights/resnet50_layer_wise/ep49_loss1.636_submit.csv',  # 10.03305
               # '../history/weights/resnet50_layer_wise_start_38/ep7_loss1.619_submit.csv',  # 10.16668 stack 6
               '../stack/resnet152_pretrain3ep_plus_6ep_8ep_bs160_ep4_loss1.508120443105214_best_submit.csv',  # 9.47133
               '../history/weights/resnet101_layer_wise/ep11_loss1.533_submit.csv',  # 9.45345
               # '../history/weights/resneXt101_234layer/ep18_loss1.54_submit.csv' # 9.92340
               '../history/weights/resneXt101_pretrain_start_18/ep6_loss1.503_submit.csv',  # 9.62907
               '../src/stack_3_fold.csv',  # 9.19270
               '../history/weights/resnet101_bs_240_0.95_30ep/finish_submit.csv',  # 9.56156
               '../history/weights/finetuning_albu/fold0_ep2_loss1.17_submit.csv'  # 9.57968
               ]

dfs_list = []

for path in submit_list:
    dfs_list.append(pd.read_csv(path))

columns = dfs_list[0].columns[1:]

for col_name in tqdm(columns):
    a = dfs_list[0][col_name]
    for df in dfs_list[1:]:
        a += df[col_name]
    a /= len(dfs_list)

    dfs_list[0][col_name] = round(a).astype('int32')

dfs_list[0].to_csv('stack_3_fold_plus_other_plus_finish_plus_train_albu.csv', index=False)

print('vse')

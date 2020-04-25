import pandas as pd
from tqdm import tqdm

submit_list = ['../stack/resnet152_pretrain3ep_plus_6ep_bs160_ep8_loss1.5298880638505576_best_submit.csv',
               '../stack/resnet101_pretrain_bs240_ep14_loss1.527_submit.csv',
                '../stack/resnet50_pretrain_bs350_submit.csv',
               '../history/weights/resnet50_layer_wise/ep49_loss1.636_submit.csv',
               '../stack/resnet152_pretrain3ep_plus_6ep_8ep_bs160_ep4_loss1.508120443105214_best_submit.csv']

df_0 = pd.read_csv(submit_list[0])
df_1 = pd.read_csv(submit_list[1])
df_2 = pd.read_csv(submit_list[2])
df_3 = pd.read_csv(submit_list[3])
df_4 = pd.read_csv(submit_list[4])

columns = df_0.columns[1:]

for col_name in tqdm(columns):
    df_0[col_name] = round((df_0[col_name] + df_1[col_name] +
                            df_2[col_name] + df_3[col_name] +
                            df_4[col_name])/5).astype('int32')

df_0.to_csv('stack_5.csv', index=False)

print('vse')

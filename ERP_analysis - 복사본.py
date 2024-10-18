# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:39:15 2024

@author: PC
"""

"""

r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\eegdata'


msid = 날짜, 이름, trial로 구성되어야함
stroop_score_db에 msid2를 참조하고
이름 날짜 식별 후, trial 추가생성

msid의 1~6 session중, 1,6 번 session을 이용해 전후 baseline labeling



"""

#%%

"""
C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\stroopdata 
경로에서 폴더 listup
을 기준으로, iddict 생성

iddict , id 1개의 1번 폴더 위치로 가서 
예를들어
C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\stroopdata\202405241653_BMS\block1

pkl listup후, 첫번째 pkl 을 load

"""


"""
1. 각 

"""
#%% EEG data load
import sys
sys.path.append(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis')
import eeg_load
import os
import glob
from tqdm import tqdm 
from collections import defaultdict
import pickle
import numpy as np
import matplotlib.pyplot as plt

directory_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\eegdata'
eeg_df = eeg_load.main(directory_path)


#%% trial 나누기

# 지정된 경로
base_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\stroopdata'

# 실험 데이터를 저장할 dict
experiment_data = defaultdict(list)

# 폴더 목록을 listup
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# 각 폴더에 대해 처리
# folder = folders[-2]
for folder in folders:
    # 고유 ID 추출 (첫 8자리 숫자 + 뒤 영어 스펠링)
    experiment_id = folder[:8] + '_' + folder[13:]
    
    # session 번호 추출 (앞의 전체 숫자)
    session_number = int(folder[:12])
    
    # 각 block의 폴더 경로를 trial로 매핑
    for block in os.listdir(os.path.join(base_path, folder)):
        block_path = os.path.join(base_path, folder, block)
        if os.path.isdir(block_path):
            if 'block1' in block:
                trial_number = 1
            elif 'block2' in block:
                trial_number = 2
            else:
                continue  # block1, block2 외의 폴더는 제외
            trial_id = f'trial{len(experiment_data[experiment_id]) + trial_number}'
            experiment_data[experiment_id].append(block_path)

# 오름차순 정렬
for key in experiment_data:
    experiment_data[key].sort()
    
#%
import os

keylist = list(experiment_data.keys())


# # 특정 경로를 설정합니다.
# base_path = r"C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\stroopdata"

# # 하위 폴더들을 리스트업합니다.
# subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# 최종 결과를 저장할 리스트를 초기화합니다.
all_pkl_files = []
for j in range(len(keylist)):
    plist = experiment_data[keylist[j]]
    for j2 in range(len(plist)):
        block_path = plist[j2]
        if os.path.exists(block_path):
            pkl_files = [os.path.join(block_path, f) for f in os.listdir(block_path) if f.endswith('.pkl')]
            # pkl_file = pkl_files[0]
            for k in range(len(pkl_files)):
                all_pkl_files.append({
                    'msid': keylist[j],
                    'trial': j2,
                    'file_path': pkl_files[k],
                    'inter_trial': k
                })

# # 결과를 출력합니다.
# print(f"총 {len(all_pkl_files)}개의 *.pkl 파일이 발견되었습니다.")
# for item in all_pkl_files:
#     print(f"Subfolder: {item['subfolder']}, Block: {item['block']}, File: {item['file_path']}")


#%%
SR = 250
tix = eeg_df[:,-1]
template = np.arange(-0.2, 1, 1/100)
X, Z = [], []
mssave = []
for i in tqdm(range(len(all_pkl_files))):
    with open(all_pkl_files[i]['file_path'], 'rb') as file:
        msdict = pickle.load(file)

    s = msdict['cue_time_stamp'] - 0.2
    e = msdict['cue_time_stamp'] + 1.5
    
    vix = np.where(np.logical_and((tix > s), (tix < e)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    
    xtmp = np.array(eeg_df[vix2][:,:2])
    if xtmp.shape[0] > 420:
        xin = []
        for ch in [0,1]:
            xtmp2 = np.array(xtmp[:420,ch])
            xtmp3 = np.array(xtmp2 / np.mean(np.abs(xtmp2)))
            xtmp4 = xtmp3 - np.mean(xtmp3[int(SR*0.2):int(SR*0.45)])
            xin.append(xtmp4)
        mssave.append(np.transpose(np.array(xin)))

        if msdict['Correct'] and msdict['Response']=='Yes': iscorrect = '맞음'
        elif not(msdict['Correct']) and msdict['Response']=='No': iscorrect = '맞음'
        elif msdict['Response'] is None: iscorrect = '응답 안함'
        else: iscorrect = '틀림'
        Z.append([msdict['Condition'], iscorrect, msdict['Response Time'], \
                  all_pkl_files[i]['msid'], all_pkl_files[i]['trial'], all_pkl_files[i]['inter_trial']])

mssave = np.array(mssave)
X, Z = np.array(mssave), np.array(Z)
xaxis = np.linspace(-200, ((420/SR)-0.2)*1000 , 420)


#%%

"""

Z.shape
Out[262]: (1665, 6)

다음과 같은 기준으로 데이터를 카테고라이즈 하여 최종적으로는 분류된 데이터를 평균처리해 하나로 만들겁니다.
첫번째 기준: Z에서, 0부터 컬럼을 카운팅할때, 3 컬럼에 존재하는 msid를 기준으로 데이터를 분류 예를들어, '20240524_BMS'
두번째 기준: 4번 컬럼을 inter_trial라고 부릅니다. 한  msid당 0~59까지 총 60개 존재합니다. 0~19, 20~39, 40~59로 
20개씩 묶어서 카테고라이즈 한다음 그 20개를 평균내어 하나로 만들어주세요.

"""
X2, Z2 = [], [] 
types = ('neutral', 'congruent', 'incongruent')
import numpy as np

label_info = np.load(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\label_info.npy')

unique_msid = np.unique(Z[:, 3])
results = []
# msid = unique_msid[0]
for msid in unique_msid:
    # Filter rows for current msid
    msid_rows = Z[Z[:, 3] == msid]
    msid_rows_X = X[Z[:, 3] == msid]
    for t in range(6):
        msid_rows_trial = msid_rows[msid_rows[:,4]==str(t)]
        msid_rows_trial_X = msid_rows_X[msid_rows[:,4]==str(t)]
        # category = (0, 19)
        for ix, category in enumerate([(0, 19), (20, 39), (40, 59)]):
            # Filter rows within current 'inter_trial' range
            and1 = np.logical_and(np.array(msid_rows_trial[:, 5], dtype=float) >= category[0],\
                np.array(msid_rows_trial[:, 5], dtype=float) <= category[1])
            and2 = np.logical_and(and1, msid_rows_trial[:,1] == '맞음')
            vix = np.where(and2)[0]
            
            if len(np.where(label_info[:,0]==msid)[0]) > 0:
                X2.append(np.mean(msid_rows_trial_X[vix], axis=(0,2)))
                mean_rt = np.mean(np.array(msid_rows_trial[vix,2], dtype=float))
                vi = np.where(label_info[:,0]==msid)[0][0]
                Z2.append([msid, t, types[ix], mean_rt, label_info[vi,1]])
            
X2, Z2 = np.array(X2), np.array(Z2)

#%%
tn = str(0)
# c0 = np.where((Z2[:,2] == 'neutral'))[0]
# c1 = np.where((Z2[:,2] == 'congruent'))[0]
# c2 = np.where((Z2[:,2] == 'incongruent'))[0]

c0 = np.where(np.logical_and((Z2[:,2] == 'neutral'), (Z2[:,1] == tn)))[0]
c1 = np.where(np.logical_and((Z2[:,2] == 'congruent'), (Z2[:,1] == tn)))[0]
c2 = np.where(np.logical_and((Z2[:,2] == 'incongruent'), (Z2[:,1] == tn)))[0]


plt.plot(xaxis, np.mean(np.array(X2)[c0], axis=0), label='neutral' + tn)
plt.plot(xaxis, np.mean(np.array(X2)[c1], axis=0), label='congruent' + tn)
plt.plot(xaxis, np.mean(np.array(X2)[c2], axis=0), label='incongruent' + tn)
plt.legend()
#%%

plt.figure(figsize=(12, 8))
colors1 = plt.cm.Blues(np.linspace(0.3, 1, 6))  # Blue shades for group 1
colors2 = plt.cm.Reds(np.linspace(0.3, 1, 6))   # Red shades for group 2

for t in range(6):
    tn = str(t)
    # c0 = np.where(np.logical_and((Z2[:,2] == 'neutral'), (Z2[:,1] == tn)))[0]
    # c1 = np.where(np.logical_and((Z2[:,2] == 'congruent'), (Z2[:,1] == tn)))[0]
    
    
    # c2 = np.logical_and((Z2[:,2] == 'incongruent'), (Z2[:,1] == tn))
    
    c1 = np.logical_and((Z2[:,2] == 'congruent'), (Z2[:,1] == tn))
    c1_vns = np.where(np.logical_and(c1, (Z2[:,4] == 'VNS')))[0]
    c1_sham = np.where(np.logical_and(c1, (Z2[:,4] == 'sham')))[0]
    
    c2 = np.logical_and((Z2[:,2] == 'incongruent'), (Z2[:,1] == tn))
    c2_vns = np.where(np.logical_and(c2, (Z2[:,4] == 'VNS')))[0]
    c2_sham = np.where(np.logical_and(c2, (Z2[:,4] == 'sham')))[0]
    
    # plt.figure()
    # plt.plot(xaxis, np.mean(np.array(X2)[c0], axis=0), label='neutral' + tn)
    # plt.plot(xaxis, np.mean(np.array(X2)[c1], axis=0), label='congruent' + tn)
    
    plt.plot(xaxis, np.mean(np.array(X2)[c1_vns], axis=0), label='congruent_VNS_' + tn, color=colors1[t])
    plt.plot(xaxis, np.mean(np.array(X2)[c1_sham], axis=0), label='congruent_sham_' + tn, color=colors2[t])
    
    # plt.plot(xaxis, np.mean(np.array(X2)[c2_vns], axis=0), label='incongruent_VNS_' + tn, color=colors1[t])
    # plt.plot(xaxis, np.mean(np.array(X2)[c2_sham], axis=0), label='incongruent_sham_' + tn, color=colors2[t])
    
plt.legend()

#%% mssave에 value 입력
import sys;
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction

trial, group = 6, 2
mssave = msFunction.msarray([trial, group])
for t in range(6):
    tn = str(t)
    s = int(SR*(0.2+0.300))
    e = int(SR*(0.2+0.450))
    
    # c0 = np.where(np.logical_and((Z2[:,2] == 'neutral'), (Z2[:,1] == tn)))[0]
    # c1 = np.where(np.logical_and((Z2[:,2] == 'congruent'), (Z2[:,1] == tn)))[0]
    
    c1 = np.logical_and((Z2[:,2] == 'congruent'), (Z2[:,1] == tn))
    # c1 = np.logical_and((Z2[:,2] == 'congruent'), (Z2[:,1] == tn))
    c1_vns = np.where(np.logical_and(c1, (Z2[:,4] == 'VNS')))[0]
    c1_sham = np.where(np.logical_and(c1, (Z2[:,4] == 'sham')))[0]
    
    c2 = np.logical_and((Z2[:,2] == 'incongruent'), (Z2[:,1] == tn))
    # c2 = np.logical_and((Z2[:,2] == 'incongruent'), (Z2[:,1] == tn))
    c2_vns = np.where(np.logical_and(c2, (Z2[:,4] == 'VNS')))[0]
    c2_sham = np.where(np.logical_and(c2, (Z2[:,4] == 'sham')))[0]
    

    tarray = np.array(X2)[c1_vns][:, s:e] - np.array(X2)[c2_vns][:, s:e]
    mssave[t][0] = np.mean(tarray, axis=1)

    tarray = np.array(X2)[c1_sham][:, s:e] - np.array(X2)[c2_sham][:, s:e]
    mssave[t][1] = np.mean(tarray, axis=1)

#%%

c0 = np.where(Z2[:,2] == 'neutral')[0]
c1 = np.where(Z2[:,2] == 'congruent')[0]
c2 = np.where(Z2[:,2] == 'incongruent')[0]

corr1 = []
for i in c2:
    if int(Z2[i][1]) > -1:
        s = int(SR*(0.2+0.300))
        e = int(SR*(0.2+0.450))
        n250_500 = np.min(X2[i][s:e])
        
        if n250_500 < -0.2:
            # n250_500_peak = np.min(X2[i][s:e])
            rt = float(Z2[i][3])
            corr1.append([n250_500, rt])
corr1 = np.array(corr1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Pearson 상관 계수 계산
corr, p_value = pearsonr(corr1[:, 0], corr1[:, 1])
print(corr)
plt.scatter(corr1[:, 0], corr1[:, 1])

#%%

c0 = np.where(Z[:,0] == 'neutral')[0]
c1 = np.where(Z[:,0] == 'congruent')[0]
c2 = np.where(Z[:,0] == 'incongruent')[0]

corr1 = []
for i in c1:
    if int(Z[i][4]) > -1 and Z[i][1] == '맞음':
        s = int(SR*(0.2+0.300))
        e = int(SR*(0.2+0.400))
        minix =  np.argsort(np.mean(X[i], axis=1)[s:e])[:5]
        n250_500 = np.mean(np.mean(X[i], axis=1)[s:e][minix])
        
        # if n250_500 < -0.0:
            # n250_500_peak = np.min(X2[i][s:e])
        rt = float(Z[i][2])
        corr1.append([n250_500, rt])
corr1 = np.array(corr1)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Pearson 상관 계수 계산
corr, p_value = pearsonr(corr1[:, 0], corr1[:, 1])
print(corr)
plt.scatter(corr1[:, 0], corr1[:, 1])


#%%



plt.xlabel('n450')
plt.ylabel('rt')
plt.title('n450 vs rt 산점도')
# 그리드 추가
plt.grid(True)
# 상관 계수 표시
plt.annotate(f"Pearson 상관 계수: {corr:.3f}", xy=(0.8, 0.8), xycoords='axes fraction')

# 그래프 표시
plt.show()

#%% ML regression
vix2 = np.concatenate((c1,c2), axis=0)
# vix2 = np.array(c2)

mlx, mly = X2[vix2], np.array(Z2[vix2,3], dtype=float)

# Slicing the data as specified
slice_1 = mlx[:, 50:112]
slice_2 = mlx[:, 112:190]

# Calculate the mean for each slice
m1 = np.mean(slice_1, axis=1)
m2 = np.mean(slice_2, axis=1)

# Calculate -(m2) / np.abs(m1)
mlx2 = (m1 - m2)*-1
positive_values = mlx2[mlx2 < 0]


# Calculate mean and standard deviation
mean = np.mean(positive_values)
std = np.std(positive_values)

# Define the threshold for outliers
lower_threshold = mean - 3 * std
upper_threshold = mean + 3 * std

# Remove values that are outside the threshold
vix3 = (positive_values >= lower_threshold) & (positive_values <= upper_threshold)

mlx3 = positive_values[vix3]
mly3 = mly[mlx2 < 0][vix3]

corr, p_value = pearsonr(mlx3, mly3)
print(corr)
plt.scatter(mlx3, mly3)
#%%



import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
x_data = np.array(mlx3)
y_data = np.array(mly3)

# Define the model
model = Sequential([
    Dense(1, input_shape=(1,))  # Single node for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
# Train the model
model.fit(x_data, y_data, epochs=50000, batch_size=2**6)



# Save the model
model.save('simple_regression_model.h5')

# Display the model summary

weights = model.get_weights()

# Print the weights
print("Weights:", weights)
plt.plot(np.array(weights[0])[:,0])

#%%

c0 = np.where((Z[:,0] == 'neutral'))[0]
c1 = np.where((Z[:,0] == 'congruent'))[0]
c2 = np.where((Z[:,0] == 'incongruent'))[0]

ch = 0

plt.figure()
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c0], axis=0), label='neutral')
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c1], axis=0), label='congruent')
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c2], axis=0), label='incongruent')
plt.legend()

plt.figure()
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c0], axis=0), label='neutral')
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c1], axis=0), label='congruent')
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c2], axis=0), label='incongruent')
plt.legend()



#%%
c0 = np.where(np.logical_and((Z[:,0] == 'neutral'), (Z[:,1] == '맞음')))[0]
c1 = np.where(np.logical_and((Z[:,0] == 'congruent'), (Z[:,1] == '맞음')))[0]
c2 = np.where(np.logical_and((Z[:,0] == 'incongruent'), (Z[:,1] == '맞음')))[0]

plt.figure()
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c0], axis=0), label='neutral')
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c1], axis=0), label='congruent')
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c2], axis=0), label='incongruent')
plt.legend()

plt.figure()
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c0], axis=0), label='neutral')
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c1], axis=0), label='congruent')
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c2], axis=0), label='incongruent')
plt.legend()

#%%
c0 = np.where(np.logical_and((Z[:,0] == 'neutral'), (Z[:,1] == '틀림')))[0]
c1 = np.where(np.logical_and((Z[:,0] == 'congruent'), (Z[:,1] == '틀림')))[0]
c2 = np.where(np.logical_and((Z[:,0] == 'incongruent'), (Z[:,1] == '틀림')))[0]

plt.figure()
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c0], axis=0), label='neutral')
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c1], axis=0), label='congruent')
plt.plot(xaxis, np.nanmedian(np.mean(np.array(X), axis=2)[c2], axis=0), label='incongruent')
plt.legend()

plt.figure()
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c0], axis=0), label='neutral')
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c1], axis=0), label='congruent')
plt.plot(xaxis, np.nanmean(np.mean(np.array(X), axis=2)[c2], axis=0), label='incongruent')
plt.legend()



#%%

xtmp = np.array(np.mean(np.array(X), axis=2)[c1])

# 임의로 100개의 데이터 추출
random_idx = np.random.choice(xtmp.shape[0], 100)
data_100 = xtmp[random_idx, :]

# 추출된 데이터의 평균값 계산
mean_data_100 = np.median(data_100, axis=0)

# 개별 데이터 시각화 (투명도 설정)
for i in range(30):
    plt.plot(xaxis, data_100[i], alpha=0.5)

# 평균값 시각화 (굵은 선)
plt.plot(xaxis, mean_data_100, linewidth=2, color='red')

# 축 라벨 및 제목 설정
plt.xlabel('열')
plt.ylabel('값')
plt.title('임의 추출된 100개 데이터와 평균값')
# 그리드 추가
# plt.grid(True)
plt.show()


plt.figure()
plt.plot(xaxis, mean_data_100, linewidth=2, color='red')



























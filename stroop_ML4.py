# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 15:27:21 2024

@author: PC
"""

"""

1. 개별 or session
2. ERP 200~350 ms
3. bandpower - alpha? -100 ~ 500 ms 으로 예측가능한지?

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
import msanalysis
import sys;
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction

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



#%% Y 추출

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_file(file_info, tix, eeg_df, SR=250):
    with open(file_info['file_path'], 'rb') as file:
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

        result_mssave = np.transpose(np.array(xin))

        if msdict['Correct'] and msdict['Response'] == 'Yes': iscorrect = '맞음'
        elif not msdict['Correct'] and msdict['Response'] == 'No': iscorrect = '맞음'
        elif msdict['Response'] is None: iscorrect = '응답 안함'
        else: iscorrect = '틀림'
        
        result_Z = [msdict['Condition'], iscorrect, msdict['Response Time'], 
                    file_info['msid'], file_info['trial'], file_info['inter_trial']]
        
        return result_mssave, result_Z
    return None, None

def main():
    SR = 250
    tix = eeg_df[:,-1]
    template = np.arange(-0.2, 1, 1/100)
    mssave = []
    Z = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_info, tix, eeg_df, SR) for file_info in all_pkl_files]
        for future in tqdm(futures):
            result = future.result()
            if result[0] is not None:
                mssave.append(result[0])
                Z.append(result[1])

    mssave = np.array(mssave)
    X, Z = np.array(mssave), np.array(Z)
    xaxis = np.linspace(-200, ((420/SR)-0.2)*1000 , 420)
    return X, Z, xaxis

X, Y, xaxis = main()

#%%
#% # 개별 데이터 기준, 맞는것만, trial 순서로, cue에 대해, group 비교
Y_subject = []
X_chmean = np.mean(X, axis=2)

label_info = np.load(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\label_info.npy', allow_pickle=True)
Z_group = []
for i in range(len(Y)):
    msid = Y[i][3]
    gix = np.where(label_info[:,0] == msid)[0]
    if len(gix) > 0:
        Z_group.append(list(Y[i]) + [label_info[gix[0], 1]])
    else:
        Z_group.append(list(Y[i]) + ['nogroup'])
Z_group = np.array(Z_group)
print(Z_group[100])

import sys
sys.path.append(r'E:\EEGstroop_exposition\Strooptask\utility')
import stroop_score_calgen
stimuluss = ['neutral', 'congruent', 'incongruent']

# msid_list = list(set(Z_group[:,3]))

ingroup = np.logical_or((Z_group[:,6] == 'VNS'), (Z_group[:,6] == 'sham'))
msid_ingroup_list = list(set(Z_group[ingroup,3]))

for i in range(len(msid_ingroup_list)):
    msid = msid_ingroup_list[i]
    vix = np.where(Z_group[:,3] == msid)[0]
    
    correct_ix = (Z_group[vix][:,1] == '맞음')
    rts = np.array(Z_group[vix][:,2], dtype=float)
    
    c0 = rts > 150
    c1 = rts < 1200
    c4 = Z_group[vix][:,4] == '1'
    
    # s = stimuluss[0]
    tmp1, tmp2 = [], []
    for s in stimuluss:
        c2 = (Z_group[vix][:,0] == s)
    
        conditions = [c0, c1, c2, c4, correct_ix]
        vix2 = np.where(np.logical_and.reduce(conditions))[0]
        
        meanrt = np.mean(rts[vix2])
        meanacc = np.mean(Z_group[vix2][:,1] == '맞음')
        tmp1.append(meanrt); tmp2.append(meanacc)
        
    
    data = np.transpose(np.array([tmp1 + tmp2]))
    score, z_scores = \
        stroop_score_calgen.stroop_tscore(data, weights = [[0.5, 0.5], [0.2, 0.3, 0.5]])
        
    Y_subject.append(data[0][0])

Y_subject = np.array(Y_subject)


#%%
tix = eeg_df[:,-1]
X = np.array(eeg_df[:,[0,1]])

psd_pre_save, psd_post_save =  [], []
# idlist = list(experiment_data.keys())

X_pre, X_post, Z = [], [], []

# id_i = 0
for id_i in tqdm(range(len(msid_ingroup_list))):
    msid = msid_ingroup_list[id_i]
    trial = 0
    directory_path = experiment_data[msid][trial]
    # .pkl 파일 목록을 저장할 리스트
    pkl_files = []
    # 지정된 경로 내의 모든 .pkl 파일을 listup
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    
    # .pkl 파일 목록 출력
    # print(pkl_files)
    pkl_files.sort()
    with open(pkl_files[0], 'rb') as file:
        msdict = pickle.load(file)
        first_cue = msdict['cue_time_stamp']
        
    trial = 5
    directory_path = experiment_data[msid][trial]
    # .pkl 파일 목록을 저장할 리스트
    pkl_files = []
    # 지정된 경로 내의 모든 .pkl 파일을 listup
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    
    # print(pkl_files)
    pkl_files.sort()
    
    with open(pkl_files[-1], 'rb') as file:
        msdict = pickle.load(file)
        last_cue = msdict['cue_time_stamp']
        
    # print(id_i)
    # print(last_cue - first_cue)
    
    # pre-baseline
    s = first_cue - (6*60)
    e = first_cue - (1*60)
    
    vix = np.where(np.logical_and((tix > s), (tix < e)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    X2 = np.transpose(np.array(X[vix2]))
    
    power, phase = msanalysis.msmain(EEGdata_ch_x_time = X2, SR=250)
    X_pre.append([power, phase])

    power2 = np.nanmedian(power, axis=1)[:,1:40]
    psd_pre = power2 / np.transpose(np.array([np.sum(power2, axis=1)]))
    psd_pre = np.mean(psd_pre, axis=0)
    # post-baseline
    s = last_cue + (0.5*60)
    e = last_cue + (5.5*60)
    
    vix = np.where(np.logical_and((tix > s), (tix < e)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    X2 = np.transpose(np.array(X[vix2]))
    
    power, phase = msanalysis.msmain(EEGdata_ch_x_time = X2, SR=250)
    X_post.append([power, phase])
    Z.append(msid)

#%% X에서 nan 제거
X_pre_nan = []

first_nan = []
for i in range(len(X_pre)):
    first_nan.append(np.where(np.isnan(np.mean(X_pre[i][0], axis=(0,1))))[0][0])
nanstartix = np.min(first_nan)
print('nanstartix', nanstartix)

X_pre_nan = []
for i in range(len(X_pre)):
    X_pre_nan.append(np.array(X_pre[i])[:,:,:,:nanstartix])
X_pre_nan = np.array(X_pre_nan)
print('X_pre_nan.shape', X_pre_nan.shape)
# shape: (pa, fn, ch, fi, ti)
#%

"""
5분 시계열로, intergrated score frist session 예측
"""

X3 = np.array(X_pre_nan)[:,0,:,:,:]
nmr = np.sum(X3, axis=2)
nmr2 = nmr[:,:,np.newaxis,:]
X3_nmr = X3 / nmr2
X3_nmr_chmean = np.mean(X3_nmr, axis=1)

target = np.array(X3_nmr_chmean)
new_size = target.shape[2] // 100
new_shape = target.shape[:2] + (new_size, 100)
arr_reshaped = target.reshape(new_shape)
arr_downsampled = np.median(arr_reshaped, axis=3)
arr_downsampled = arr_downsampled[:,:,:,np.newaxis]
input_shape = arr_downsampled.shape[1:]

#%%
# plt.plot(arr_downsampled[0,:,310,0])


#%% parallel CNN

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold

# Example data
# Xt = np.random.rand(376, 50, 150, 1)  # Replace with your actual data
# Yt = np.random.rand(376)  # Replace with your actual data

# Define the model architecture

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_regression_cnn_model(input_shape):
    # 입력 레이어
    inputs = Input(shape=input_shape)
    
    # 첫 번째 합성곱 블록
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 두 번째 합성곱 블록
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 세 번째 합성곱 블록
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 완전 연결 레이어
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # 회귀 출력을 위한 출력 레이어
    outputs = Dense(1)(x)  # 활성화 함수 없음 (선형 활성화)
    
    # 모델 정의
    model = Model(inputs=inputs, outputs=outputs)
    
    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model


# Example usage
input_shape = arr_downsampled.shape[1:]  # Assuming you downsample the data by a factor of 10
model = create_regression_cnn_model(input_shape)


# input_shape = (72500, 50, 2)
# model = create_parallel_model(input_shape)
model.summary()

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1

Xt, Yt = np.array(arr_downsampled), np.array(Y_subject)
Yt = Yt / 690
# nmr = np.mean(Xt, axis=(0,1,3,4))
# nmr_expanded = nmr[None, None, :, None, None]

# Xt = Xt / 450000

#%%
import pickle

# Xt와 Yt를 pkl 파일로 저장하기
if False:
    with open(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel2\data.pkl', 'wb') as f:
        pickle.dump({'Xt': Xt, 'Yt': Yt, 'Z_group': Z_group}, f)

# 저장한 pkl 파일에서 Xt와 Yt를 로드하기
with open(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel2\data.pkl', 'rb') as f:
    data = pickle.load(f)
    Xt = data['Xt']
    Yt = data['Yt']
    Z_group = data['Z_group']

# 로드된 데이터 확인
print('Xt_loaded shape:', Xt.shape)
print('Yt_loaded shape:', Yt.shape)

#%%
val_loss_list = []
for cv in range(len(Xt)):
    tlist = list(range(len(Xt)))
    telist = [cv]
    trlist = list(set(tlist) - set(telist))
    X_train, X_val = Xt[trlist], Xt[telist]
    Y_train, Y_val = Yt[trlist], Yt[telist]
    
    # X_train = [X_train[:,0,:,:,:], X_train[:,1,:,:,:]]
    # X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
    
    model = create_regression_cnn_model(input_shape)
    history = model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_val, Y_val))
    val_loss_list.append(history.history['val_loss'])

print('Cross-validation training completed.')
#%%
val_loss_list = np.array(val_loss_list)
plt.plot(np.mean(val_loss_list, axis=0))

#%% post data 분석

# model
model_total = create_regression_cnn_model(input_shape)
history = model_total.fit(Xt, Yt, epochs=1000, batch_size=32)

# msid = msid_ingroup_list[0]
groups = []
for msid in msid_ingroup_list:
    vix = np.where(Z_group[:,3] == msid )[0][0]
    groups.append(Z_group[vix][-1])
groups = np.array(groups)
groups2 = []
group_match = []

# X data
X_post_nan = []
first_nan = []
for i in range(len(X_post)):
    first_nan.append(np.where(np.isnan(np.mean(X_post[i][0], axis=(0,1))))[0][0])
nanstartix = np.min(first_nan)
print('nanstartix', 72500)
nanstartix = 72500

X_post_nan = []
for i in range(len(X_post)):
    tmp = np.array(X_post[i])[:,:,:,:nanstartix]
    if not(np.isnan(np.mean(tmp))):
        X_post_nan.append(tmp)
        groups2.append(groups[i])
        group_match.append(i)
X_post_nan = np.array(X_post_nan)
print('X_post_nan.shape', X_post_nan.shape)

X3 = np.array(X_post_nan)[:,0,:,:,:]
nmr = np.sum(X3, axis=2)
nmr2 = nmr[:,:,np.newaxis,:]
X3_nmr = X3 / nmr2
X3_nmr_chmean = np.mean(X3_nmr, axis=1)

target = np.array(X3_nmr_chmean)
new_size = target.shape[2] // 100
new_shape = target.shape[:2] + (new_size, 100)
arr_reshaped = target.reshape(new_shape)
arr_downsampled_post = np.median(arr_reshaped, axis=3)
arr_downsampled_post = arr_downsampled_post[:,:,:,np.newaxis]
print(arr_downsampled_post.shape)
# input_shape = arr_downsampled.shape[1:]

# predict
yhat_pre = model_total.predict(arr_downsampled)
yhat_pre2 = model_total.predict(Xt)
yhat = model_total.predict(arr_downsampled_post)


groups2 = np.array(groups2)
g0 = np.where(groups2=='sham')[0]
g1 = np.where(groups2=='VNS')[0]
arr1, arr2 = yhat[:,0][g0], yhat[:,0][g1]
print(np.mean(arr1), np.mean(arr2))

import numpy as np
from scipy import stats
t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")



groups = np.array(groups)
g0 = np.where(groups=='sham')[0]
g1 = np.where(groups=='VNS')[0]
arr1_pre, arr2_pre = yhat_pre[:,0][g0], yhat_pre[:,0][g1]
t_stat, p_value = stats.ttest_ind(arr1_pre, arr2_pre, equal_var=False)

print(np.mean(arr1), np.mean(arr2))
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")


groups2 = np.array(groups2)
g0 = np.where(groups2=='sham')[0]
g1 = np.where(groups2=='VNS')[0]
arr1, arr2 = yhat[:,0][g0], yhat[:,0][g1]
print(np.mean(arr1), np.mean(arr2))

group_match = np.array(group_match)
pre_yhat = Yt[group_match]
post_yhat = yhat[:,0]

plt.scatter(pre_yhat[g0], post_yhat[g0])
plt.scatter(pre_yhat[g1], post_yhat[g1])

print(np.mean(pre_yhat[g0]), np.mean(post_yhat[g0]))
print(np.mean(pre_yhat[g1]), np.mean(post_yhat[g1]))

arr1 = post_yhat[g0] - pre_yhat[g0]
arr2 = post_yhat[g1] - pre_yhat[g1]
t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")
print(np.mean(arr1), np.mean(arr2))



















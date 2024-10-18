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

#%%

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
            # xtmp3 = np.array(xtmp2 / np.mean(np.abs(xtmp2)))
            # xtmp4 = xtmp3 - np.mean(xtmp3[int(SR*0.2):int(SR*0.45)])
            xin.append(xtmp2)

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

X, Z, xaxis = main()

#%% data preprocessing

# Z_group
label_info = np.load(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\label_info.npy', allow_pickle=True)
Z_group = []
for i in range(len(Z)):
    msid = Z[i][3]
    gix = np.where(label_info[:,0] == msid)[0]
    if len(gix) > 0:
        Z_group.append(list(Z[i]) + [label_info[gix[0], 1]])
    else:
        Z_group.append(list(Z[i]) + ['nogroup'])
Z_group = np.array(Z_group)
print(Z_group[100])

# X morlet wavelet
X2 = np.array(X)
power_ch0, phase_ch0 = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,0], SR=250)
power_ch1, phase_ch1 = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,1], SR=250)


SR = 250
s, e = int((0.2)*SR), int((0.5+0.2)*SR)

X3 = np.array([[power_ch0[:,:,s:e], power_ch1[:,:,s:e]], [phase_ch0[:,:,s:e], phase_ch1[:,:,s:e]]]) # fn x ch x n x fi x t
# X3 = X3.transpose(1,0,2,3)

#%% session 분리

Xml, Zml = [], []

ingroup = np.logical_or(Z_group[:,-1]=='sham', Z_group[:,-1]=='VNS')
subject_list = list(set(Z_group[:,3]))
for s in range(len(subject_list)):
    # idix = Z_group[:,3]==subject_list[s]
    # baseix = Z_group[:,4]=='0'
    
    # conditions = [ingroup, idix, baseix]
    # vix = np.where(np.logical_and.reduce(conditions))[0]  
    
    # tmp = np.median(X3[vix][:, :, :, s:e], axis=3)
    # tmp2 = tmp / np.sum(tmp, axis=2)[:, :, np.newaxis]
    # tmp3 = np.mean(tmp2 , axis=1)
    # base_fi = np.mean(tmp3, axis=0)
    
    idix = Z_group[:,3]==subject_list[s]
    baseix = np.logical_or(Z_group[:,4]=='1', Z_group[:,4]=='2')
    conditions = [ingroup, idix, baseix]
    vix2 = np.where(np.logical_and.reduce(conditions))[0]  
    
    # i = vix2[0]
    for i in vix2:
        X3_tlimit = np.array(X3[:,:,i,:,:])
        X3_axis = X3_tlimit.transpose(0,3,2,1)
        
        tmpallo = np.zeros(X3_axis[0].shape) * np.nan
        for fi in range(50):
            for ch in range(2):
                tmpallo[:,fi,ch] = X3_axis[0,:,fi,ch] / np.sum(X3_axis[0,:,fi,ch])
        X3_axis[0] = tmpallo
        
        Xml.append(X3_axis)
        Zml.append(Z_group[i])
        
Xml = np.array(Xml); Zml = np.array(Zml)       
print(Xml.shape)
Yml = np.array(Zml[:,1] == '맞음', dtype=int)

#%%

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from scipy import stats


ingroup = np.logical_or((Z_group[:,6] == 'sham'), (Z_group[:,6] == 'VNS'))
incorrect = (Z_group[:,1] == '맞음')
conditions = [ingroup, incorrect]
vix = np.where(np.logical_and.reduce(conditions))[0]

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr, arr_min, arr_max

def create_parallel_model(input_shape):
    # 첫 번째 입력 및 CNN
    input1 = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation='relu')(input1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(128, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Flatten()(x1)
    
    # 두 번째 입력 및 CNN
    # input2 = Input(shape=input_shape)
    # x2 = Conv2D(32, (3, 3), activation='relu')(input2)
    # x2 = MaxPooling2D((2, 2))(x2)
    # x2 = Dropout(0.25)(x2)
    
    # x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    # x2 = MaxPooling2D((2, 2))(x2)
    # x2 = Dropout(0.25)(x2)
    
    # x2 = Conv2D(128, (3, 3), activation='relu')(x2)
    # x2 = MaxPooling2D((2, 2))(x2)
    # x2 = Dropout(0.25)(x2)
    
    # x2 = Flatten()(x2)
    
    # 병합 및 최종 출력 레이어
    # merged = concatenate([x1, x2])
    merged = Dense(64, activation='relu')(x1)
    merged = Dropout(0.5)(merged)
    output = Dense(1, activation='sigmoid')(merged)  # 회귀를 위한 출력 레이어
    
    model = Model(inputs=[input1], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_parallel_model(input_shape):
    # 첫 번째 입력 및 CNN
    input1 = Input(shape=input_shape)
    x1 = Conv2D(32, (5, 5), activation='relu')(input1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.1)(x1)
    
    x1 = Conv2D(32, (5, 5), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.1)(x1)

    x1 = Flatten()(x1)
    
    # 병합 및 최종 출력 레이어
    # merged = concatenate([x1, x2])
    merged = Dense(32, activation='relu')(x1)
    merged = Dropout(0.1)(merged)
    output = Dense(1, activation='sigmoid')(merged)  # 회귀를 위한 출력 레이어
    
    model = Model(inputs=[input1], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
#%%

Xml = np.array(Xml); Zml = np.array(Zml)       
print(Xml.shape)
Yml = np.array(Zml[:,1] == '맞음', dtype=int)

input_shape = Xml.shape[2:]
model = create_parallel_model(input_shape)


X_train = np.array(Xml)
X_train = np.array(X_train[:,0,:,:,:])
history = model.fit(X_train, Yml, epochs=500, batch_size=32)

spath = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel\\'

if False: model.save_weights(spath + f'model_fold_test.h5')

#%%

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential


kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10개의 폴드로 나누기

# Cross-validation 루프
for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
    print(f"Fold {fold+1}")
    
    # Train/Validation 셋을 나눕니다
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    Y_train_fold, Y_val_fold = Yml[train_index], Yml[val_index]
    
    # 모델 생성
    model = create_parallel_model(input_shape)
    history = model.fit(X_train_fold, Y_train_fold, epochs=500, batch_size=32, 
                        validation_data=(X_val_fold, Y_val_fold))
    
    # 필요에 따라 모델 저장, 성능 평가 등을 추가로 할 수 있습니다.

#%%

from sklearn.metrics import roc_curve, roc_auc_score

model.load_weights(spath + f'model_fold_test.h5')

frequency_interval = {'delta': [0, 4], \
                      'theta': [4, 8], \
                      'alpha': [8, 13], \
                      'beta': [13, 30], \
                      'gamma': [30, 40]}

    
pix = np.where(Yml==1)[0]
nix = np.where(Yml==0)[0]


mssave = msFunction.msarray([5])

keylist = list(frequency_interval.keys())
forlist = np.arange(-4,  3.1, 0.1)
for i in range(len(keylist)):
    for m in forlist:
        fi_s = frequency_interval[keylist[i]][0]
        fi_e = frequency_interval[keylist[i]][1]
    
        X_train = np.array(Xml)
        X_train_dummy = np.array(X_train[:,0,:,:,:])
        scale = np.mean(X_train_dummy[:,:,fi_s:fi_e,:])
        X_train_dummy[:,:,fi_s:fi_e,:] = X_train_dummy[:,:,fi_s:fi_e,:] + (scale * m)
        
        yhat = model.predict(X_train_dummy)
        
        # fpr, tpr, thresholds = roc_curve(Yml, yhat)
    
        # # 최적의 임계값 계산
        # optimal_idx = np.argmax(tpr - fpr)
        # optimal_threshold = thresholds[optimal_idx]
        
        # # print(f"Optimal Threshold: {optimal_threshold:.4f}")
        
        # # 최적 임계값에서 negative로 예측된 샘플 수 계산
        # yhat_optimal = (yhat < optimal_threshold).astype(int)
        # negative_predictions = np.sum(yhat_optimal)
        
        # print(f"Number of negative predictions at optimal threshold: {negative_predictions}")
        
        # AUC 계산
        auc = roc_auc_score(Yml, yhat)
        
        # print(keylist[i], f"AUC: {auc:.4f}")
        # print('sum', np.sum(yhat<0.5))
        
        mssave[i].append(auc)


for i in range(5):
    plt.plot(forlist, mssave[i], label=keylist[i])
plt.legend()








#%%

spath = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel\\'

train_losses = []
val_losses = []

# if not(os.path.isfile(spath + f'model_fold_{fold_no}.h5')):
for train_index, val_index in kf.split(Xt):
    print(f'Training fold {fold_no} ...')
    
    X_train, X_val = Xt[train_index], Xt[val_index]
    Y_train, Y_val = Yt[train_index], Yt[val_index]
    
    X_train = [X_train[:,0,:,:,:], X_train[:,1,:,:,:]]
    X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
    
    model = create_parallel_model(input_shape)
    
    # # Early stopping and model checkpoint callbacks
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # model_checkpoint = ModelCheckpoint(spath + f'model_fold_{fold_no}.h5', save_best_only=True, monitor='val_loss')
    
    # # Train the model
    # history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val), 
    #                     callbacks=[early_stopping, model_checkpoint])
    
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val))
    model.save_weights(spath + f'model_fold_{fold_no}.h5')

    # Append losses
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])
    
    fold_no += 1

print('Cross-validation training completed.')











#%%


# plt.plot(np.median(Xml[Yml], axis=0))
# plt.plot(np.median(Xml[np.array(Yml==0, dtype=int)], axis=0))

import numpy as np
import matplotlib.pyplot as plt

# Yml이 1인 경우의 데이터와 0인 경우의 데이터를 구분

g0 = np.where(Zml[:,-1]=='VNS')[0]
g1 = np.where(Zml[:,-1]=='sham')[0]

g0 = np.where(Zml[:,1]=='맞음')[0]
g1 = np.where(Zml[:,1]=='틀림')[0]

group0 = Xml[g0]
group1 = Xml[g1]

# 각 그룹에 대한 중앙값 계산
median_group1 = np.mean(group1, axis=0)
median_group0 = np.mean(group0, axis=0)

# SEM (표준 오차) 계산
sem_group1 = np.std(group1, axis=0) / np.sqrt(group1.shape[0])
sem_group0 = np.std(group0, axis=0) / np.sqrt(group0.shape[0])

# 플롯 생성
plt.figure(figsize=(10, 6))

# Group 1 (Yml == 1)
plt.plot(median_group1, label='Group 1 (Yml == 1)')
plt.fill_between(range(len(median_group1)), 
                 median_group1 - sem_group1, 
                 median_group1 + sem_group1, 
                 color='blue', alpha=0.2)

# Group 0 (Yml == 0)
plt.plot(median_group0, label='Group 0 (Yml == 0)')
plt.fill_between(range(len(median_group0)), 
                 median_group0 - sem_group0, 
                 median_group0 + sem_group0, 
                 color='orange', alpha=0.2)

# 라벨, 타이틀, 범례 추가
plt.xlabel('Feature Index')
plt.ylabel('Median Value')
plt.title('Median with SEM Shaded')
plt.legend()
plt.show()

#%%

# 이거 morlet wavlet으로 예측하고,
# beta power 늘리면 어케됨?

#%%

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 모델 정의
model = Sequential()

# 첫 번째 은닉층
model.add(Dense(64, input_shape=(Xml.shape[1],), activation='relu'))
# model.add(Dropout(0.5))  # 과적합 방지를 위해 드롭아웃 추가

# 두 번째 은닉층
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.5))

# 출력층 (이진 분류이므로 출력 뉴런은 1개, 활성화 함수는 시그모이드)
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 모델 요약 출력
model.summary()

# 모델 학습
history = model.fit(Xml, Yml, epochs=500, batch_size=32, validation_split=0.2, verbose=1)

# 모델 평가
loss, accuracy = model.evaluate(Xml, Yml)
print(f"Final loss: {loss:.4f}")
print(f"Final accuracy: {accuracy:.4f}")





#%%

from scipy.stats import sem

# print(Z_group[0])

fi = {'delta': [1, 4], \
      'theta': [4, 8], \
      'alpha': [8, 13], \
      'beta': [13, 30], \
      'gamma': [30, 50]}

colors1 = plt.cm.Blues(np.linspace(0.3, 1, 3))  # Blue shades for group 1
colors2 = plt.cm.Reds(np.linspace(0.3, 1, 3))   # Red shades for group 2
cues = ['neutral', 'congruent', 'incongruent']

SR = 250
# s, e = int((0)*SR), int((0+0.2)*SR)
s, e = int((0.2)*SR), int((0.5+0.2)*SR)
# s, e = 0, int((1.5+0.2)*SR)
xaxis = np.linspace(-200, ((420/SR)-0.2)*1000 , 420)
sessions = [[0,1],[2,3],[4,5]]
vns_PSD, sham_PSD = [], []

# cues = ['neutral', 'congruent', 'incongruent']
# # ERPsave = {}
# for cue in cues:
    
plt.figure()
for t in range(len(sessions)):
    c0 = np.logical_or(np.array(Z_group[:,4], dtype=float) == sessions[t][0],\
                       np.array(Z_group[:,4], dtype=float) == sessions[t][1])
    
    g_vns = (Z_group[:,-1] == 'VNS')
    g_sham = (Z_group[:,-1] == 'sham')
    
    conditions = [c0, g_vns]
    vix = np.where(np.logical_and.reduce(conditions))[0]  
    # fi_power_vns = np.mean(X3[vix][:, :, fi[power][0]:fi[power][1], s:e], axis=(1,2)) 
    tmp = np.median(X3[vix][:, :, :, s:e], axis=3)
    tmp2 = tmp / np.sum(tmp, axis=2)[:, :, np.newaxis]
    fi_power_vns = np.mean(tmp2 , axis=1)
    vns_PSD.append(fi_power_vns)
    
    # plt.plot(np.mean(tmp3, axis=0))

    conditions = [c0, g_sham]
    vix2 = np.where(np.logical_and.reduce(conditions))[0]
    # fi_power_sham = np.mean(X3[vix2][:, :, fi[power][0]:fi[power][1], :], axis=(1,2))
    tmp = np.median(X3[vix2][:, :, :, s:e], axis=3)
    tmp2 = tmp / np.sum(tmp, axis=2)[:, :, np.newaxis]
    fi_power_sham = np.mean(tmp2 , axis=1)
    sham_PSD.append(fi_power_sham)
    
















#%%

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from scipy import stats


ingroup = np.logical_or((Z_group[:,6] == 'sham'), (Z_group[:,6] == 'VNS'))
incorrect = (Z_group[:,1] == '맞음')
conditions = [ingroup, incorrect]
vix = np.where(np.logical_and.reduce(conditions))[0]

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    return normalized_arr, arr_min, arr_max

def create_parallel_model(input_shape):
    # 첫 번째 입력 및 CNN
    input1 = Input(shape=input_shape)
    x1 = Conv2D(32, (3, 3), activation='relu')(input1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(64, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Conv2D(128, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.25)(x1)
    
    x1 = Flatten()(x1)
    
    # 두 번째 입력 및 CNN
    input2 = Input(shape=input_shape)
    x2 = Conv2D(32, (3, 3), activation='relu')(input2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.25)(x2)
    
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.25)(x2)
    
    x2 = Conv2D(128, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.25)(x2)
    
    x2 = Flatten()(x2)
    
    # 병합 및 최종 출력 레이어
    merged = concatenate([x1, x2])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(1)(merged)  # 회귀를 위한 출력 레이어
    
    model = Model(inputs=[input1, input2], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    
    return model

#%%


frequency_interval = {'delta': [0, 4], \
                      'theta': [4, 8], \
                      'alpha': [8, 13], \
                      'beta': [13, 30], \
                      'gamma': [30, 40]}
    
# -0.25~1.5 -> -0.25 부터 0.5까지 50 ms 단위로 예측
# 각 frequency 별로 예측

timeline_as_ms = np.arange(-250, 500, 40) + 250
timeline_as_sr = np.array(timeline_as_ms * SR/1000, dtype=int)
# print(timeline_as_sr)

loss_tsave = []
# for fi in frequency_interval:

Xt, Yt = [], []
for i in vix:
    # x1 = np.mean(X3_chmean_fi[i], axis=0)
    X3_tlimit = np.array(X3[:,:,i,:,0:int(SR*(0.25+0.5))])
    X3_axis = X3_tlimit.transpose(0,3,2,1)
    
    tmpallo = np.zeros(X3_axis[0].shape) * np.nan
    for fi in range(50):
        for ch in range(2):
            tmpallo[:,fi,ch] = X3_axis[0,:,fi,ch] / np.sum(X3_axis[0,:,fi,ch])
    
    X3_axis[0] = tmpallo
    
    # X3_axis2 = X3_axis[:,:, sf:ef ,:]
    # X3_axis2 = X3_axis[:, t:t+10, : ,:]
    
    
    rt = float(Z_group[i,2])
    Xt.append(X3_axis)
    Yt.append(rt)

# from sklearn.preprocessing import StandardScaler

Xt, Yt = np.array(Xt), np.array(Yt)
print(Xt.shape, Yt.shape)

# 배열 정규화
# input_data1 = Xt[:]  # 예시 입력 데이터 1
# input_data2 = Xt[1]  # 예시 입력 데이터 2

# scaler = StandardScaler()
# scaled_input_data1 = scaler.fit_transform(input_data1.reshape(-1, 2)).reshape(input_data1.shape)
# scaled_input_data2 = scaler.fit_transform(input_data2.reshape(-1, 2)).reshape(input_data2.shape)
# Xt = np.array([scaled_input_data1, scaled_input_data2])

Yt, nmr_max, nmr_min = normalize(Yt)
print(Yt)


#%%
#% parallel CNN

input_shape = Xt.shape[2:]
model = create_parallel_model(input_shape)
model.summary()

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1

spath = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel\\'

train_losses = []
val_losses = []

# if not(os.path.isfile(spath + f'model_fold_{fold_no}.h5')):
for train_index, val_index in kf.split(Xt):
    print(f'Training fold {fold_no} ...')
    
    X_train, X_val = Xt[train_index], Xt[val_index]
    Y_train, Y_val = Yt[train_index], Yt[val_index]
    
    X_train = [X_train[:,0,:,:,:], X_train[:,1,:,:,:]]
    X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
    
    model = create_parallel_model(input_shape)
    
    # # Early stopping and model checkpoint callbacks
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # model_checkpoint = ModelCheckpoint(spath + f'model_fold_{fold_no}.h5', save_best_only=True, monitor='val_loss')
    
    # # Train the model
    # history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val), 
    #                     callbacks=[early_stopping, model_checkpoint])
    
    history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_val, Y_val))
    model.save_weights(spath + f'model_fold_{fold_no}.h5')

    # Append losses
    train_losses.append(history.history['loss'])
    val_losses.append(history.history['val_loss'])
    
    fold_no += 1

print('Cross-validation training completed.')

#%%

output_file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result_data\learning_curve.pkl'
if False:
    msdict = {'train_losses': train_losses, 'val_losses': val_losses}
    with open(output_file_path, 'wb') as file:
        pickle.dump(msdict, file)
        
with open(output_file_path, 'rb') as file:
    msdict = pickle.load(file)
    
train_losses = msdict['train_losses']
val_losses = msdict['val_losses']

# Calculate mean and SEM for train and validation losses
mean_train_loss = np.mean(train_losses, axis=0)
sem_train_loss = np.std(train_losses, axis=0) / np.sqrt(len(train_losses))

mean_val_loss = np.mean(val_losses, axis=0)
sem_val_loss = np.std(val_losses, axis=0) / np.sqrt(len(val_losses))

# Plotting the results
epochs = range(1, len(mean_train_loss) + 1)

plt.figure(figsize=(4, 3))

# Plot train losses
plt.plot(epochs, mean_train_loss, label='Mean Training Loss', color='#779ECB')
plt.fill_between(epochs, mean_train_loss - sem_train_loss, mean_train_loss + sem_train_loss, color='#AEC6CF', alpha=0.8)

# Plot validation losses
plt.plot(epochs, mean_val_loss, label='Mean Test Loss', color='#FF6961')
plt.fill_between(epochs, mean_val_loss - sem_val_loss, mean_val_loss + sem_val_loss, color='#FFB3BA', alpha=0.8)

plt.title('Learning Curves with Mean and SEM')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()

if False:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure5_learning_curve.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

#%% predict
fold_no = 1
predictions = []
ground_truths = []
losses = []
val_ix_save = []

for train_index, val_index in kf.split(Xt):
    _, X_val = Xt[train_index], Xt[val_index]
    _, Y_val = Yt[train_index], Yt[val_index]
    X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
    
    best_model = create_parallel_model(input_shape)
    
    # Load the best model for this fold
    # best_model = load_model(spath + f'model_fold_{fold_no}.h5')
    best_model.load_weights(spath + f'model_fold_{fold_no}.h5')

    # Make predictions on the validation set
    preds = best_model.predict(X_val)
    loss = best_model.evaluate(X_val, Y_val, verbose=0)
    
    # Store the predictions and ground truths
    predictions.extend(preds)
    ground_truths.extend(Y_val)
    val_ix_save.extend(val_index)
    losses.append(loss)
    
    fold_no += 1

# Convert predictions and ground truths to numpy arrays for further analysis
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)
val_ix_save = np.array(val_ix_save)

print('loss value', np.mean(losses))
print('Cross-validation and prediction completed.')
# loss_tsave.append([t, np.mean(losses)])

# plt.scatter(predictions[:,0], ground_truths, s=5, alpha=0.3)
# correlation, p_value = pearsonr(predictions[:,0], ground_truths)
# print(f"Pearson correlation coefficient: {correlation}")
# print(f"P-value: {p_value}")
    

#%% trial mean


stimulus_types = np.array(['neutral', 'congruent', 'incongruent'])
ingroup = np.logical_or((Z_group[:,6] == 'sham'), (Z_group[:,6] == 'VNS'))
incorrect = (Z_group[:,1] == '맞음')

vixsave, vixz = [], []
group_yhat_y = []
mskey_list = list(set(Z_group[:,3]))
for i in range(len(mskey_list)):
    for j in range(len(stimulus_types)):
        for se in range(6):
            stimulus = stimulus_types[j]
            msid = mskey_list[i]
            
            c0 = (Z_group[:,3] == msid)
            c1 = (Z_group[:,0] == stimulus)
            c2 = (Z_group[:,4] == str(se))
            
            conditions = [c0, c1, c2, ingroup, incorrect]
            vix = np.where(np.logical_and.reduce(conditions))[0]
            
            tvix = []
            for k in vix:
                wix = np.where(val_ix_save==k)[0]
                if len(wix) > 0:
                    tvix.append(wix[0])
            if len(tvix) > 0:        
                predictions_group = predictions[tvix,0]
                ground_truths_group = ground_truths[tvix]
                
                nan_indices_pred = np.isnan(predictions_group)
                nan_indices_gt = np.isnan(ground_truths_group)
                nan_indices_combined = nan_indices_pred | nan_indices_gt
                # NaN이 없는 위치의 데이터만 저장
                filtered_predictions = predictions_group[~nan_indices_combined]
                filtered_ground_truths = ground_truths_group[~nan_indices_combined]
                
                # if np.isnan(np.mean(filtered_predictions)):
                #     print(i, j, se)
                        
                group_yhat_y.append([np.mean(filtered_predictions), np.mean(filtered_ground_truths)])


group_yhat_y = np.array(group_yhat_y)

def minmax_restore(arr):
    arr_min = nmr_min
    arr_max = nmr_max
    return arr * (arr_max - arr_min) + arr_min

scatter_x =  minmax_restore(group_yhat_y[:,0])
scatter_y =  minmax_restore(group_yhat_y[:,1])

plt.scatter(scatter_x, scatter_y, alpha=0.5, edgecolors='none')
# 1차 다항식 회귀 (직선 추세선) 계산
coefficients = np.polyfit(scatter_x, scatter_y, 1)
polynomial = np.poly1d(coefficients)
trendline = polynomial(scatter_x)
plt.plot(scatter_x, trendline, color='#FFA07A', linewidth=2, alpha=0.5)


if False:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure5_scatter.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()


correlation, p_value = stats.pearsonr(group_yhat_y[:,0], group_yhat_y[:,1])
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")


#%% figure5 - power

vix = np.argsort(Yt)[:int(len(Yt)*0.01)]
ptmp = np.transpose(np.mean(Xt[vix[[3]], 0, :, :, 0], axis=0))

plt.figure(figsize=(3, 3))  # 정사각형 크기 설정
plt.imshow(ptmp, cmap='twilight', aspect='auto')

# x축 레이블 설정
x_positions = range(0, data.shape[1], 30)
x_labels = np.array(np.round(np.linspace(-250, 500, len(x_positions)), -1), dtype=int)
plt.xticks(ticks=x_positions, labels=x_labels)

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure5_power.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)

plt.show()

#%% figure5 - phase
 
ptmp = np.transpose(np.mean(Xt[vix[[3]], 1, :, :, 0], axis=0))

plt.figure(figsize=(3, 3))  # 정사각형 크기 설정
plt.imshow(ptmp, cmap='twilight', aspect='auto')

# x축 레이블 설정
x_positions = range(0, data.shape[1], 30)
x_labels = np.array(np.round(np.linspace(-250, 500, len(x_positions)), -1), dtype=int)
plt.xticks(ticks=x_positions, labels=x_labels)

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure5_phase.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import pickle

if False:
    msdict_figure4 = {
    'epochs': epochs,
    'mean_train_loss': mean_train_loss,
    'sem_train_loss': sem_train_loss,
    'mean_val_loss': mean_val_loss,
    'sem_val_loss': sem_val_loss,
    'scatter_x': scatter_x,
    'scatter_y': scatter_y,
    'trendline': trendline}

spath = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure' + '\\'

if False:
    with open(spath + 'vnstroop_figure4.pickle', 'wb') as file:
        pickle.dump(msdict_figure4, file)
    
with open(spath + 'vnstroop_figure4.pickle', 'rb') as file:
    msdict = pickle.load(file)

epochs = msdict['epochs']
mean_train_loss = msdict['mean_train_loss']
sem_train_loss = msdict['sem_train_loss']
scatter_x = msdict['scatter_x']
scatter_y = msdict['scatter_y']

color1 = '#7fdada'
color2 = '#FF6961'
color3 = '#e1c58b'

#%% figure4 subplot

def set_custom_yticks(ylim, num_divisions=6, roundn=3, offset=0):
    """ylim을 6분할하여 5개의 tick을 설정하는 함수"""
    ymin, ymax = ylim
    tick_interval = (ymax - ymin) / num_divisions
    yticks = np.round(np.arange(ymin, ymax * 1.001, tick_interval)[1:-1] + offset, roundn)
    return yticks

fig, axs = plt.subplots(1, 2, figsize=(4.36/2 * 3 * 1, 2.45/2 * 2 * 1.4 / 2))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

## A

alpha = 0.2
axs[0].plot(epochs, mean_train_loss, label='Mean Training Loss', c=color1, lw=1)
axs[0].fill_between(epochs, mean_train_loss - \
                       sem_train_loss, mean_train_loss + sem_train_loss, color=color1, alpha=0.8)

axs[0].plot(epochs, mean_val_loss, label='Mean Test Loss', c=color2, lw=1)
axs[0].fill_between(epochs, mean_val_loss - \
                       sem_val_loss, mean_val_loss + sem_val_loss, color=color2, alpha=0.8)

axs[0].set_ylim(0.00, 0.06)
axs[0].set_yticks(set_custom_yticks((0.00, 0.057+0.0001), num_divisions=5, offset=-0.001))
axs[0].set_xlim(-5, 105)
axs[0].set_xticks(np.arange(0, 100 + 1, 20))
axs[0].set_ylabel('Mean Squared Error', fontsize=7, labelpad=0.1)
axs[0].set_xlabel('Epochs', fontsize=7, labelpad=0.5)

#%
## B

axs[1].scatter(scatter_x, scatter_y, alpha=0.5, edgecolors='none', c=color1, s=20)
axs[1].plot(scatter_x, trendline, color=color2, linewidth=2, alpha=0.5)

ylim = (np.round(np.min(scatter_y* 0.98)), np.round(np.max(scatter_y) * 1.02))
xlim = (np.round(np.min(scatter_x* 0.98)), np.round(np.max(scatter_x) * 1.02))

axs[1].set_ylim(ylim)
axs[1].set_yticks(np.arange(650, 1000, 50))
axs[1].set_xlim(xlim)
axs[1].set_xticks(np.arange(730, 950, 50))
axs[1].set_ylabel('Ground Truth RT (ms)', fontsize=7, labelpad=0.1)
axs[1].set_xlabel('Estimated RT (ms)', fontsize=7, labelpad=0.5)

for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=7, pad=2, width=0.2)  # x tick 폰트 크기 7로 설정
    ax.tick_params(axis='y', labelsize=7, pad=0.2, width=0.2)  # y tick 폰트 크기 7로 설정
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure4.png'
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()




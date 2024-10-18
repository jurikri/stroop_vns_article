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

X, Z, xaxis = main()

#%% # 개별 데이터 기준, 맞는것만, trial 순서로, cue에 대해, group 비교
X_chmean = np.mean(X, axis=2)

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

#%%
X2 = np.array(X)
power_ch0, phase_ch0 = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,0], SR=250)
power_ch1, phase_ch1 = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,1], SR=250)


X3 = np.array([[power_ch0, power_ch1], [phase_ch0, phase_ch1]]) # fn x ch x n x fi x t
SR = 250




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
    lo
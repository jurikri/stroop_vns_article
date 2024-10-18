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
    return normalized_arr

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
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

def create_parallel_model(input_shape):
    # 첫 번째 입력 및 CNN
    n = 2
    input1 = Input(shape=input_shape)
    x1 = Conv2D(64*n, (3, 3), activation='relu')(input1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Conv2D(128*n, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Conv2D(256*n, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Conv2D(512*n, (3, 3), activation='relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Flatten()(x1)
    
    # 두 번째 입력 및 CNN
    input2 = Input(shape=input_shape)
    x2 = Conv2D(64*n, (3, 3), activation='relu')(input2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv2D(128*n, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv2D(256*n, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv2D(512*n, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Flatten()(x2)
    
    # 병합 및 최종 출력 레이어
    merged = concatenate([x1, x2])
    merged = Dense(256*n, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(128*n, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    output = Dense(1)(merged)  # 회귀를 위한 출력 레이어
    
    model = Model(inputs=[input1, input2], outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
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
    
    # X3_axis2 = X3_axis[:,:, sf:ef ,:]
    # X3_axis2 = X3_axis[:, t:t+10, : ,:]
    
    
    rt = float(Z_group[i,2])
    Xt.append(X3_axis)
    Yt.append(rt)

from sklearn.preprocessing import StandardScaler

Xt, Yt = np.array(Xt), np.array(Yt)
print(Xt.shape, Yt.shape)

# 배열 정규화
# input_data1 = Xt[:]  # 예시 입력 데이터 1
# input_data2 = Xt[1]  # 예시 입력 데이터 2


from sklearn.preprocessing import RobustScaler

Xt_nmr = np.zeros(Xt.shape)*np.nan
for axe0 in tqdm(range(Xt.shape[1])):
    for axe1 in range(Xt.shape[2]):
        for axe2 in range(Xt.shape[3]):
            for axe3 in range(Xt.shape[4]):   
                selected_data = Xt[:, axe0, axe1, axe2, axe3].reshape(-1, 1)
                scaler = RobustScaler()
                Xt_nmr[:, axe0, axe1,axe2 ,axe3] = scaler.fit_transform(selected_data)[:,0]

Xt = np.array(Xt_nmr)
Yt = normalize(Yt)

#% parallel CNN
#%%
input_shape = Xt.shape[2:]
model = create_parallel_model(input_shape)
model.summary()

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1

spath = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\MLmodel\\'

for train_index, val_index in kf.split(Xt):
    weight_savename = spath + f'model_fold_{fold_no}.h5'
    if not(os.path.isfile(weight_savename)):
        print(f'Training fold {fold_no} ...')
        
        X_train, X_val = Xt[train_index], Xt[val_index]
        Y_train, Y_val = Yt[train_index], Yt[val_index]
        
        X_train = [X_train[:,0,:,:,:], X_train[:,1,:,:,:]]
        X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
        
        model = create_parallel_model(input_shape)
        
        # Early stopping and model checkpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=1000, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(weight_savename, save_best_only=True, monitor='val_loss')
        
        # Train the model
        model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_val, Y_val), 
                  callbacks=[early_stopping, model_checkpoint])
        
        fold_no += 1

print('Cross-validation training completed.')

#% predict


fold_no = 1
predictions = []
ground_truths = []
losses = []
val_ix_save = []

for train_index, val_index in kf.split(Xt):
    _, X_val = Xt[train_index], Xt[val_index]
    _, Y_val = Yt[train_index], Yt[val_index]
    X_val  = [X_val [:,0,:,:,:], X_val [:,1,:,:,:]]
    
    # model = create_parallel_model(input_shape)
    
    # Load the best model for this fold
    best_model = load_model(spath + f'model_fold_{fold_no}.h5')
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

plt.scatter(group_yhat_y[:,0], group_yhat_y[:,1])
correlation, p_value = stats.pearsonr(group_yhat_y[:,0], group_yhat_y[:,1])
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")


#%%

integrated_grad1_save = []


for fold_no in tqdm(range(10,11)):

# fold_no = 1
    best_model = load_model(spath + f'model_fold_{fold_no}.h5')
    
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    
    @tf.function
    def compute_gradients(model, inputs, target_index):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = model(inputs)
            loss = predictions[:, target_index]
        grads = tape.gradient(loss, inputs)
        return grads
    
    def integrated_gradients(model, inputs, baseline=None, num_steps=50):
        if baseline is None:
            baseline = [np.zeros_like(input) for input in inputs]
        assert all(b.shape == i.shape for b, i in zip(baseline, inputs))
        
        interpolated_inputs = []
        for i in range(len(inputs)):
            interpolated_input = []
            for alpha in np.linspace(0, 1, num_steps + 1):
                interpolated_input.append(baseline[i] + alpha * (inputs[i] - baseline[i]))
            interpolated_inputs.append(interpolated_input)
        
        avg_gradients = []
        for i in range(len(inputs)):
            interpolated_grads = []
            for step in range(num_steps + 1):
                step_inputs = [tf.convert_to_tensor(interpolated_inputs[j][step], dtype=tf.float32) for j in range(len(inputs))]
                grads = compute_gradients(model, step_inputs, 0)
                interpolated_grads.append(grads[i])
            avg_gradients.append(tf.reduce_mean(interpolated_grads, axis=0).numpy())
        
        integrated_gradients = [(inputs[i] - baseline[i]) * avg_gradients[i] for i in range(len(inputs))]
        return integrated_gradients
    
    def plot_importance(importance, title):
        plt.figure(figsize=(10, 8))
        plt.imshow(importance, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()
    
    # input_data1 = Xt[:,0]  # 예시 입력 데이터 1
    # input_data2 = Xt[:,1]  # 예시 입력 데이터 2
    
    # baseline_data1 = np.zeros_like(input_data1)
    # baseline_data2 = np.zeros_like(input_data2)
    
    # integrated_grads = integrated_gradients(best_model, [input_data1, input_data2], baseline=[baseline_data1, baseline_data2])
    
    batch_size = 64  # 배치 크기를 줄입니다.
    # 데이터셋을 작은 배치로 나눠서 처리합니다.
    for batch_start in tqdm(range(0, len(Xt), batch_size)):
        batch_end = batch_start + batch_size
        input_data1_batch = Xt[batch_start:batch_end, 0]
        input_data2_batch = Xt[batch_start:batch_end, 1]
        baseline_data1_batch = np.zeros_like(input_data1_batch)
        baseline_data2_batch = np.zeros_like(input_data2_batch)
    
        integrated_grads = integrated_gradients(best_model, [input_data1_batch, input_data2_batch], baseline=[baseline_data1_batch, baseline_data2_batch])
    
        integrated_grad1 = integrated_grads[0][0]
        integrated_grad2 = integrated_grads[1][0]
    
        integrated_grad1_save.append(integrated_grad1)
    
    # integrated_grad1 = integrated_grads[0][0]  # 첫 번째 입력의 기울기
    # integrated_grad2 = integrated_grads[1][0]  # 두 번째 입력의 기울기
    
    # plot_importance(integrated_grad1[:, :, 0], 'Importance of Input 1, Channel 0')
    # plot_importance(integrated_grad1[:, :, 1], 'Importance of Input 1, Channel 1')
    # plot_importance(integrated_grad2[:, :, 0], 'Importance of Input 2, Channel 0')
    # plot_importance(integrated_grad2[:, :, 1], 'Importance of Input 2, Channel 1')
    
    # integrated_grad1_save.append(integrated_grad1)

#%%

def moving_average_smoothing(yvalues, ws):
    # Ensure the window size is valid
    if ws < 1:
        raise ValueError("Window size must be at least 1")
    
    # Create an array to store the smoothed values
    smoothed_values = np.zeros_like(yvalues)
    
    # Pad the array at the beginning and end to handle edge cases
    padded_yvalues = np.pad(yvalues, (ws//2, ws//2), mode='edge')
    
    # Compute the moving average
    for i in range(len(yvalues)):
        smoothed_values[i] = np.mean(padded_yvalues[i:i+ws])
    
    return smoothed_values

integrated_grad1_save = np.array(integrated_grad1_save)
ptmp = np.mean(np.mean(np.median(integrated_grad1_save, axis=0), axis=2), axis=1)
ptmp2 = moving_average_smoothing(ptmp, 20)
taxis = np.linspace(-250, 500, len(ptmp2))
plt.plot(taxis, ptmp2)

#%%

frequency_interval = {'delta': [0, 4], \
                      'theta': [4, 8], \
                      'alpha': [8, 13], \
                      'beta': [13, 30], \
                      'gamma': [30, 50]}

integrated_grad1_save = np.array(integrated_grad1_save)
ptmp = np.mean(np.mean(np.median(integrated_grad1_save, axis=0), axis=2), axis=0)
ptmp2 = moving_average_smoothing(ptmp, 1)
plt.plot(ptmp)

for mskey in frequency_interval:
    s,e = frequency_interval[mskey]
    print(mskey, np.mean(ptmp[s:e]))


#%%

integrated_grad1_save = np.array(integrated_grad1_save)
ptmp = np.mean(np.median(integrated_grad1_save, axis=0), axis=2)
ptmp2 = np.transpose(ptmp)
# plt.imshow(ptmp2)

ptmp3 = []
for mskey in frequency_interval:
    s,e = frequency_interval[mskey]
    fiplot = np.mean(ptmp2[s:e, :], axis=0)
    fiplot2 = moving_average_smoothing(fiplot, 20)
    plt.figure()
    plt.plot(taxis, fiplot2, label=mskey)
    plt.legend()
    
#%%


"""
1. theta, alpha의 100~200 ms의 negativity
2. beta 100~150 ms의 negativity
3. beta의 위를 제외한 전체적인 positivity가 rt와 corr?
"""

plt.plot(timeline_as_ms - 250, np.array(loss_tsave)[:,1])



Xt_chmean_power = np.mean(Xt, axis=4)[:,0]
result = Xt_chmean_power * ptmp

bandpower = np.mean(result, axis=(1,2))


vix = np.where(bandpower < 0.000000000001)[0]

p1, p2 = bandpower[vix], Yt[vix]
plt.scatter(p1, p2, s=5, alpha=0.3)
correlation, p_value = stats.pearsonr(p1, p2)
print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")


#%% 시각화

taxis = np.linspace(-250, 500, int(SR*(0.25+0.5)))
yplot = np.mean(X2, axis=(0,2))[0:int(SR*(0.25+0.5))]

plt.plot(taxis, yplot)


fi_time = np.median(X3[0,:], axis=(0,1))
plt.imshow(fi_time)




















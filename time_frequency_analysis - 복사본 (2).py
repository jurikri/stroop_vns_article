# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:39:15 2024

@author: PC
"""

"""
이 코드는 ERP에서 넘어옴
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
power_ch0, _ = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,0], SR=250)
power_ch1, _ = msanalysis.msmain(EEGdata_ch_x_time = X2[:,:,1], SR=250)
X3 = np.array([power_ch0, power_ch1])
X3 = X3.transpose(1,0,2,3)

X3_chmean = np.nanmean(X3, axis=1)

X3_chmean = np.array(power_ch0)
print(X3_chmean.shape)

#%%
from scipy.stats import sem

print(Z_group[0])

fi = {'delta': [1, 4], \
      'theta': [4, 8], \
      'alpha': [8, 13], \
      'beta': [13, 30], \
      'gamma': [30, 50]}

colors1 = plt.cm.Blues(np.linspace(0.3, 1, 6))  # Blue shades for group 1
colors2 = plt.cm.Reds(np.linspace(0.3, 1, 6))   # Red shades for group 2
cues = ['neutral', 'congruent', 'incongruent']

SR = 250
# s, e = int((0)*SR), int((0+0.2)*SR)
s, e = int((0.2)*SR), int((0.5+0.2)*SR)
# s, e = 0, int((1.5+0.2)*SR)
xaxis = np.linspace(-200, ((420/SR)-0.2)*1000 , 420)
sessions = [[0,1],[2,3],[4,5]]

# cue, t = cues[0], 0

for power in fi:
    
    means_vns = []
    errors_vns = []
    means_sham = []
    errors_sham = []
    labels = []
    plt.figure()
    for t in range(len(sessions)):
        c0 = np.logical_or(np.array(Z_group[:,4], dtype=float) == sessions[t][0],\
                           np.array(Z_group[:,4], dtype=float) == sessions[t][1])
        # c1 = (Z_group[:,0] == cue)
        # c2 = (Z_group[:,1] == '맞음')
        
        g_vns = (Z_group[:,-1] == 'VNS')
        g_sham = (Z_group[:,-1] == 'sham')
        
        conditions = [c0, g_vns]
        vix = np.where(np.logical_and.reduce(conditions))[0]  
        fi_power_vns = np.mean(np.median(X3[vix][:, :, fi[power][0]:fi[power][1], s:e], axis=3), axis=(1,2))
        
        conditions = [c0, g_sham]
        vix2 = np.where(np.logical_and.reduce(conditions))[0]
        fi_power_sham = np.mean(np.median(X3[vix2][:, :, fi[power][0]:fi[power][1], s:e], axis=3), axis=(1,2))
        
        print(power, t)
        print(np.mean(fi_power_vns), np.mean(fi_power_sham))
        
        
        fi_power_vns_subject, fi_power_sham_subject = [], []
        
        keylist = list(set(Z_group[vix,3]))
        for msid in keylist:
            kvix = np.where(Z_group[vix,3]==msid)[0]
            fi_power_vns_subject.append(np.mean(fi_power_vns[kvix]))
        
        keylist = list(set(Z_group[vix2,3]))
        for msid in keylist:
            kvix = np.where(Z_group[vix2,3]==msid)[0]
            fi_power_sham_subject.append(np.mean(fi_power_sham[kvix]))
             
        if t == 0:
            base_vns = np.array(fi_power_vns_subject)
            base_sham = np.array(fi_power_sham_subject)
            
        vnsdb = 10*np.log10(fi_power_vns_subject / base_vns)
        mean_vns = np.mean(vnsdb)
        error_vns = sem(vnsdb)
        
        shamdb = 10*np.log10(fi_power_sham_subject / base_sham)
        mean_sham = np.mean(shamdb)
        error_sham = sem(shamdb)
        
        means_vns.append(mean_vns)
        errors_vns.append(error_vns)
        means_sham.append(mean_sham)
        errors_sham.append(error_sham)
        labels.append(f"{t}")
        
    x_values = [1,2,3]
    plt.plot(x_values, means_vns, label='VNS', color='blue')
    plt.fill_between(x_values, 
                     np.array(means_vns) - np.array(errors_vns), 
                     np.array(means_vns) + np.array(errors_vns), 
                     color='blue', alpha=0.2)
    
    # Sham 데이터 그래프
    plt.plot(x_values, means_sham, label='Sham', color='red')
    plt.fill_between(x_values, 
                     np.array(means_sham) - np.array(errors_sham), 
                     np.array(means_sham) + np.array(errors_sham), 
                     color='red', alpha=0.2)
    plt.title(str(cue) + '_' + str(power))
    
        
        
    # 막대 그래프 그리기
    # x_pos = np.arange(len(labels))
    # bar_width = 0.35
    
    # plt.bar(x_pos - bar_width/2, means_vns, bar_width, yerr=errors_vns, \
    #         align='center', alpha=0.7, ecolor='black', capsize=10, label='VNS')
        
    # plt.bar(x_pos + bar_width/2, means_sham, bar_width, yerr=errors_sham, \
    #         align='center', alpha=0.7, ecolor='black', capsize=10, label='Sham')
    
    # plt.xticks(x_pos, labels)
    # plt.xlabel('Conditions')
    # plt.ylabel('Mean Value with SEM')
    # plt.title(str(cue) + '_' + str(power))
    # plt.legend()
        
            
            
            


# n450_vns = np.mean(X3_chmean[vix][:,s:e], axis=1)

"""
time-frequency analysis는 시간특이적인 차이가 관찰되지 않으므로 접고,
cue 주변 frequency power가 다른듯 하니, 여기서 의미를 찾을수 있는지 살펴보자
"""

#%%
from scipy.stats import sem

print(Z_group[0])

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

# cue, t = cues[0], 0

# for power in fi:
    
#     means_vns = []
#     errors_vns = []
#     means_sham = []
#     errors_sham = []
#     labels = []


vns_PSD, sham_PSD = [], []

plt.figure()
for t in range(len(sessions)):
    c0 = np.logical_or(np.array(Z_group[:,4], dtype=float) == sessions[t][0],\
                       np.array(Z_group[:,4], dtype=float) == sessions[t][1])
    # c1 = (Z_group[:,0] == cue)
    # c2 = (Z_group[:,1] == '맞음')
    
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
    
    
    print(power, t)
    print(np.mean(fi_power_vns), np.mean(fi_power_sham))
    
    
    plt.plot(np.mean(fi_power_vns, axis=0)[4:], c=colors1[t])
    plt.plot(np.mean(fi_power_sham, axis=0)[4:], c=colors2[t])
    plt.title(str(t)) 

#%%
msdict = {'vns_PSD': vns_PSD, 'sham_PSD': sham_PSD}
with open(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis' + '\stroop_PSD.pkl', 'wb') as file:
    pickle.dump(msdict, file)


#%% subject x session x fi = power

sessions = [[0,1],[2,3],[4,5]]
s, e = int((0.2)*SR), int((0.5+0.2)*SR)

total_keylist = np.array(list(set(Z_group[:,3])))
subject_x_se_x_fi_power = np.zeros((len(total_keylist), 3, 50)) * np.nan

for t in range(len(sessions)):
    c0 = np.logical_or(np.array(Z_group[:,4], dtype=float) == sessions[t][0],\
                       np.array(Z_group[:,4], dtype=float) == sessions[t][1])
              
    vix = np.where(c0)[0]  
        
    keylist = list(set(Z_group[vix,3]))
    for msid in keylist:
        kvix = np.where(Z_group[vix,3]==msid)[0]  
        tmp = np.median(X3[kvix][:, :, :, s:e], axis=3)
        tmp2 = tmp / np.sum(tmp, axis=2)[:, :, np.newaxis]
        fi_power = np.mean(np.mean(tmp2 , axis=1), axis=0)
        ki = np.where(total_keylist==msid)[0][0]
        subject_x_se_x_fi_power[ki, t, :] = fi_power
        
subject_x_se_x_fi_power_beta = np.mean(subject_x_se_x_fi_power[:,:,20:27], axis=2)
        
#% subject x session = tscore

subject_x_se_x_tscore = np.zeros((len(total_keylist), 3)) * np.nan
for ix, msid in enumerate(total_keylist):
    for t in range(len(sessions)):
        c0 = msdict_array[:,14] == msid
        c1 = np.logical_or((msdict_array[:,13] == sessions[t][0]), (msdict_array[:,13] == sessions[t][1]))
        conditions = [c0, c1]
        vix = np.where(np.logical_and.reduce(conditions))[0]  
        subject_x_se_x_tscore[ix, t] = np.mean(tscore_save[vix])
 
#%
from scipy.stats import pearsonr
delta_tscore1 = subject_x_se_x_tscore[:,2] - subject_x_se_x_tscore[:,0]
delta__beta1 = subject_x_se_x_fi_power_beta[:,2] - subject_x_se_x_fi_power_beta[:,0]

delta_tscore2 = subject_x_se_x_tscore[:,1] - subject_x_se_x_tscore[:,0]
delta__beta2 = subject_x_se_x_fi_power_beta[:,1] - subject_x_se_x_fi_power_beta[:,0]


delta_tscore = np.concatenate((delta_tscore1, delta_tscore2), axis=0)
delta__beta = np.concatenate((delta__beta1, delta__beta2), axis=0)

# 3 표준편차 초과하는 아웃라이어 식별
mean_beta = np.mean(delta__beta)
std_beta = np.std(delta__beta)
threshold = 4 * std_beta

# 아웃라이어 인덱스 찾기
outliers_idx = np.where(np.abs(delta__beta - mean_beta) > threshold)[0]

# 아웃라이어 제거
delta__beta_cleaned = np.delete(delta__beta, outliers_idx)
delta_tscore_cleaned = np.delete(delta_tscore, outliers_idx)


r_value, p_value = pearsonr(delta__beta_cleaned, delta_tscore_cleaned)
print(f"Pearson correlation coefficient: {r_value}")
print(f"P-value: {p_value}")

# 산점도 그리기
plt.scatter(delta__beta_cleaned, delta_tscore_cleaned)
plt.xlabel('delta__beta')
plt.ylabel('delta_tscore')
plt.title(f'Scatter Plot and Pearson r: {r_value:.2f}')
plt.grid(True)
plt.show()


#%%

psd_pre_save2, psd_post_save2 = np.array(psd_pre_save), np.array(psd_post_save)
psd_pre_save2, psd_post_save2 = psd_pre_save2[:,3:], psd_post_save2[:,3:]
label_info3 = label_info2[:len(psd_pre_save)]
g0 = np.where(label_info3=='VNS')[0]
g1 = np.where(label_info3=='sham')[0]
s_fi, e_if = 4, 40
target_freqs_05 = np.linspace(s_fi, e_if, 36)


median_psd = np.mean(psd_pre_save2[g1], axis=0)
sem_psd = sem(psd_pre_save2[g1], axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval_vns')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)

median_psd = np.mean(psd_post_save2[g1], axis=0)
sem_psd = sem(psd_pre_save2[g1], axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval_vns')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)


t = 2
target = np.array(msdict['sham_PSD'][t])[:, s_fi:e_if]
target = target / np.sum(target, axis=1)[:, np.newaxis]
median_psd = np.mean(target, axis=0)
sem_psd = sem(target, axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval_vns')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)

t = 0
target = np.array(msdict['sham_PSD'][t])[:, s_fi:e_if]
target = target / np.sum(target, axis=1)[:, np.newaxis]
median_psd = np.mean(target, axis=0)
sem_psd = sem(target, axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval_vns')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)


        #%%
        
        fi_power_vns_subject, fi_power_sham_subject = [], []
        
        keylist = list(set(Z_group[vix,3]))
        for msid in keylist:
            kvix = np.where(Z_group[vix,3]==msid)[0]
            fi_power_vns_subject.append(np.mean(fi_power_vns[kvix]))
        
        keylist = list(set(Z_group[vix2,3]))
        for msid in keylist:
            kvix = np.where(Z_group[vix2,3]==msid)[0]
            fi_power_sham_subject.append(np.mean(fi_power_sham[kvix]))
             
        if t == 0:
            base_vns = np.array(fi_power_vns_subject)
            base_sham = np.array(fi_power_sham_subject)
            
        vnsdb = 10*np.log10(fi_power_vns_subject / base_vns)
        mean_vns = np.mean(vnsdb)
        error_vns = sem(vnsdb)
        
        shamdb = 10*np.log10(fi_power_sham_subject / base_sham)
        mean_sham = np.mean(shamdb)
        error_sham = sem(shamdb)
        
        means_vns.append(mean_vns)
        errors_vns.append(error_vns)
        means_sham.append(mean_sham)
        errors_sham.append(error_sham)
        labels.append(f"{t}")
        
    x_values = [1,2,3]
    plt.plot(x_values, means_vns, label='VNS', color='blue')
    plt.fill_between(x_values, 
                     np.array(means_vns) - np.array(errors_vns), 
                     np.array(means_vns) + np.array(errors_vns), 
                     color='blue', alpha=0.2)
    
    # Sham 데이터 그래프
    plt.plot(x_values, means_sham, label='Sham', color='red')
    plt.fill_between(x_values, 
                     np.array(means_sham) - np.array(errors_sham), 
                     np.array(means_sham) + np.array(errors_sham), 
                     color='red', alpha=0.2)
    plt.title(str(cue) + '_' + str(power))



#%%
import numpy as np

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


smoothed_values = moving_average_smoothing(yvalues, ws)
print(smoothed_values)


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


colors1 = plt.cm.Blues(np.linspace(0.3, 1, 6))  # Blue shades for group 1
colors2 = plt.cm.Reds(np.linspace(0.3, 1, 6))   # Red shades for group 2

# cue = 'incongruent'

cues = ['neutral', 'congruent', 'incongruent']
# ERPsave = {}
for cue in cues:
    plt.figure(figsize=(12, 8))
    c1 = (Z_group[:,0] == cue)
    c2 = (Z_group[:,1] == '맞음')
    
    g_vns = (Z_group[:,6] == 'VNS')
    g_sham = (Z_group[:,6] == 'sham')
    
    vns_plots = []
    sham_plots = []
    
    # yvalues_vnss, yvalues_shams = [], []
    for t in range(6):
        trial_n = str(t)
        c0 = (Z_group[:,4] == trial_n)
        
        conditions = [c0, c1, c2, g_vns]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        yvalues_vns = np.mean(np.array(X_chmean)[vix], axis=0)
        yvalues_smoothing = moving_average_smoothing(yvalues_vns, 20)
        plot, = plt.plot(xaxis, yvalues_smoothing, label = cue + trial_n + '_VNS', color=colors1[t])
        vns_plots.append(plot)
        
        conditions = [c0, c1, c2, g_sham]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        yvalues_sham = np.mean(np.array(X_chmean)[vix], axis=0)
        yvalues_smoothing = moving_average_smoothing(yvalues_sham, 20)
        plot, = plt.plot(xaxis, yvalues_smoothing, label = cue + trial_n + '_SHAM', color=colors2[t])
        sham_plots.append(plot)
        
        # yvalues_vnss.append(yvalues_vns)
        # yvalues_shams.append(yvalues_sham)
        
    # ERPsave[cue] = np.array([yvalues_vnss, yvalues_shams])
        
    plots = sham_plots + vns_plots
    labels = [plot.get_label() for plot in plots]
    
    if False: # figure save codes
        plt.legend(plots, labels)
        plt.xlabel('Time (ms) from Cue')
        plt.ylabel('Relative Voltage')
        plt.title('EEG Response Relative to Cue Time')
        fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
        fsave2 = fsave + '\\' + cue + '_ERP.png'
        plt.tight_layout()
        plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
        plt.show()

#%%

"""
1. 위 코드 cue 3가지 자동화, figure save
2. 데이터 받아서 밑에 분석 진행
3. performance와 corr 분석


"""


print(Z_group[100])

cues = ['neutral', 'congruent', 'incongruent']

#%%

cue = 'incongruent'
c1 = (Z_group[:,0] == cue)
c2 = (Z_group[:,1] == '맞음')

# g_vns = (Z_group[:,6] == 'VNS')
# g_sham = (Z_group[:,6] == 'sham')

conditions = [c1, c2]
vix = np.where(np.logical_and.reduce(conditions))[0]

SR = 250
s = int(SR*(0.25 + 0.35))
e = int(SR*(0.25 + 0.5))

n450 = np.mean(X_chmean[vix][:,s:e], axis=1)
rts = np.array(Z[vix][:,2], dtype=float)

corr1 = [n450, rts]
corr1 = np.transpose(np.array(corr1))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Pearson 상관 계수 계산
corr, p_value = pearsonr(corr1[:, 0], corr1[:, 1])
print(corr)
plt.scatter(corr1[:, 0], corr1[:, 1], s=0.5)



#%% mssave에 value 입력






ERPsave[cue]

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





















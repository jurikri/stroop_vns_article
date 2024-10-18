# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:39:40 2024

@author: PC
"""

r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\eegdata'

"""
msid = 날짜, 이름, trial로 구성되어야함
stroop_score_db에 msid2를 참조하고
이름 날짜 식별 후, trial 추가생성

msid의 1~6 session중, 1,6 번 session을 이용해 전후 baseline labeling
"""



"""
Sham (Original): 밝은 청록색 (#B7FFF4)
VNS (Original): 밝은 연보라색 (#E2DCFF)
Sham (New): 진한 청록색 (#73E0D1)
VNS (New): 진한 연보라색 (#C7B6FF)
Sham (Extended): 더욱 진한 청록색 (#49C4B1)
VNS (Extended): 더욱 진한 보라색 (#A182FF)
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

import sys
sys.path.append(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis')
import eeg_load
import os
from collections import defaultdict
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import msanalysis
from tqdm import tqdm

#%%

directory_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\stroopdata' # 지정된 경로
folders = [] # 폴더 목록을 저장할 리스트
# 지정된 경로의 폴더를 listup
for item in os.listdir(directory_path):
    item_path = os.path.join(directory_path, item)
    if os.path.isdir(item_path):
        folders.append(item)

#% trial 나누기

# 지정된 경로
base_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\stroopdata'

# 실험 데이터를 저장할 dict
experiment_data = defaultdict(list)

# 폴더 목록을 listup
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# 각 폴더에 대해 처리
# folder = folders[0]
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

# 디렉토리 경로를 입력하여 메인 함수를 호출합니다
directory_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\eegdata'
eeg_df = eeg_load.main(directory_path)

beta1, beta2 = 1, 1
tix = eeg_df[:,-1]
X = np.array(eeg_df[:,[0,1]])
        

#%%

psd_pre_save, psd_post_save =  [], []
idlist = list(experiment_data.keys())
# id_i = 0
for id_i in tqdm(range(len(idlist))):
    msid = idlist[id_i]
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
    power2 = np.nanmedian(power, axis=1)[:,1:40]
    psd_post = power2 / np.transpose(np.array([np.sum(power2, axis=1)]))
    psd_post = np.mean(psd_post , axis=0)

    # plt.plot(psd_pre)
    # plt.plot(psd_post)

    psd_pre_save.append(np.array(psd_pre))
    psd_post_save.append(np.array(psd_post))


#%%
idlist = list(experiment_data.keys())
label_info = np.load(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\label_info.npy', allow_pickle=True)
label_info2 = []
for id_i in tqdm(range(len(idlist))):
    msid = idlist[id_i]
    msid2 = np.where(label_info[:,0]==msid)[0]
    if len(msid2) > 0:
        label_info2.append(label_info[msid2[0], 1])
label_info2 = np.array(label_info2)

#%% Figure3A - predata
# fig1 = plt.figure()
psd_pre_save, psd_post_save = np.array(psd_pre_save), np.array(psd_post_save)

label_info3 = label_info2[:len(psd_pre_save)]
g0 = np.where(label_info3=='VNS')[0]
g1 = np.where(label_info3=='sham')[0]

s_fi, e_if = 1, 40
target_freqs_05 = np.linspace(s_fi, e_if, psd_pre_save.shape[1])

# alpha = 0.5

psd_pre_median_VNS = np.median(psd_pre_save[g0], axis=0)
psd_pre_sem_VNS = sem(psd_pre_save[g0], axis=0)

psd_post_median_VNS = np.median(psd_post_save[g0], axis=0)
psd_post_sem_VNS = sem(psd_post_save[g0], axis=0)

psd_pre_median_sham = np.median(psd_pre_save[g1], axis=0)
psd_pre_sem_sham = sem(psd_pre_save[g1], axis=0)

psd_post_median_sham = np.median(psd_post_save[g1], axis=0)
psd_post_sem_sham = sem(psd_post_save[g1], axis=0)

## Figure3B - predata

data_3b_vns = 10*np.log10(psd_post_save[g0]/psd_pre_save[g0])
median_psd_3b_vns = np.median(data_3b_vns, axis=0)
sem_psd_3b_vns = sem(data_3b_vns, axis=0)

data_3b_sham = 10*np.log10(psd_post_save[g1]/psd_pre_save[g1])
median_psd_3b_sham = np.median(data_3b_sham, axis=0)
sem_psd_3b_sham = sem(data_3b_sham, axis=0)

## C - predata

# Frequency interval indices for different bands
theta_band = slice(4, 8)    # Example: 0-7 index for theta band (4-8 Hz)
alpha_band = slice(8, 15)   # Example: 8-14 index for alpha band (8-13 Hz)
beta_band = slice(15, 30)   # Example: 15-29 index for beta band (13-30 Hz)
gamma_band = slice(30, 39)  # Example: 30-38 index for gamma band (30-40 Hz)

# Compute mean for each frequency band
theta_g0_mean = np.mean(data_3b_vns[:, theta_band], axis=1)
alpha_g0_mean = np.mean(data_3b_vns[:, alpha_band], axis=1)
beta_g0_mean = np.mean(data_3b_vns[:, beta_band], axis=1)
gamma_g0_mean = np.mean(data_3b_vns[:, gamma_band], axis=1)

theta_g1_mean = np.mean(data_3b_sham[:, theta_band], axis=1)
alpha_g1_mean = np.mean(data_3b_sham[:, alpha_band], axis=1)
beta_g1_mean = np.mean(data_3b_sham[:, beta_band], axis=1)
gamma_g1_mean = np.mean(data_3b_sham[:, gamma_band], axis=1)

# Prepare data for bar plot
bands = ['Theta', 'Alpha', 'Beta', 'Gamma']
g0_means = [np.mean(theta_g0_mean), np.mean(alpha_g0_mean), np.mean(beta_g0_mean), np.mean(gamma_g0_mean)]
g1_means = [np.mean(theta_g1_mean), np.mean(alpha_g1_mean), np.mean(beta_g1_mean), np.mean(gamma_g1_mean)]

def compute_sem(data, band):
    return np.std(data[:, band], axis=1) / np.sqrt(data.shape[0])

theta_g0_sem = compute_sem(data_3b_vns, theta_band)
alpha_g0_sem = compute_sem(data_3b_vns, alpha_band)
beta_g0_sem = compute_sem(data_3b_vns, beta_band)
gamma_g0_sem = compute_sem(data_3b_vns, gamma_band)

theta_g1_sem = compute_sem(data_3b_sham, theta_band)
alpha_g1_sem = compute_sem(data_3b_sham, alpha_band)
beta_g1_sem = compute_sem(data_3b_sham, beta_band)
gamma_g1_sem = compute_sem(data_3b_sham, gamma_band)

# Prepare SEM data for bar plot
g0_sems = [np.mean(theta_g0_sem), np.mean(alpha_g0_sem), np.mean(beta_g0_sem), np.mean(gamma_g0_sem)]
g1_sems = [np.mean(theta_g1_sem), np.mean(alpha_g1_sem), np.mean(beta_g1_sem), np.mean(gamma_g1_sem)]


from scipy.stats import ttest_ind

# Theta band에 대한 t-test
theta_tstat, theta_pvalue = ttest_ind(theta_g0_mean, theta_g1_mean, equal_var=False)

# Alpha band에 대한 t-test
alpha_tstat, alpha_pvalue = ttest_ind(alpha_g0_mean, alpha_g1_mean, equal_var=False)

# Beta band에 대한 t-test
beta_tstat, beta_pvalue = ttest_ind(beta_g0_mean, beta_g1_mean, equal_var=False)

# Gamma band에 대한 t-test
gamma_tstat, gamma_pvalue = ttest_ind(gamma_g0_mean, gamma_g1_mean, equal_var=False)

# 결과 출력
print(f"Theta band: t-statistic = {theta_tstat}, p-value = {theta_pvalue}")
print(f"Alpha band: t-statistic = {alpha_tstat}, p-value = {alpha_pvalue}")
print(f"Beta band: t-statistic = {beta_tstat}, p-value = {beta_pvalue}")
print(f"Gamma band: t-statistic = {gamma_tstat}, p-value = {gamma_pvalue}")


#%% Figure3 D-F predata

#% EEG data load
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

#%
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
#% data preprocessing

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

#%% Figure3D

from scipy.stats import sem

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

figure3D_plot_data = []

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# plt.figure()
for t in range(len(sessions)):
    c0 = np.logical_or(np.array(Z_group[:,4], dtype=float) == sessions[t][0],
                       np.array(Z_group[:,4], dtype=float) == sessions[t][1])
    
    g_vns = (Z_group[:,-1] == 'VNS')
    g_sham = (Z_group[:,-1] == 'sham')
    
    conditions = [c0, g_vns]
    vix = np.where(np.logical_and.reduce(conditions))[0]
    tmp = np.median(X3[vix][:, :, :, s:e], axis=3)
    tmp2 = tmp / np.sum(tmp, axis=2)[:, :, np.newaxis]
    fi_power_vns = np.mean(tmp2, axis=1)
    vns_PSD.append(fi_power_vns)
    
    conditions = [c0, g_sham]
    vix2 = np.where(np.logical_and.reduce(conditions))[0]
    tmp = np.median(X3[vix2][:, :, :, s:e], axis=3)
    tmp2 = tmp / np.sum(tmp, axis=2)[:, :, np.newaxis]
    fi_power_sham = np.mean(tmp2, axis=1)
    sham_PSD.append(fi_power_sham)
    
    # 계산된 평균
    mean_vns = np.mean(fi_power_vns, axis=0)
    mean_sham = np.mean(fi_power_sham, axis=0)
    
    # SEM 계산
    sem_vns = sem(fi_power_vns, axis=0)
    sem_sham = sem(fi_power_sham, axis=0)
    
    figure3D_plot_data.append([mean_sham, mean_vns, sem_sham, sem_vns])
    

#%%
msdict = {'vns_PSD': vns_PSD, 'sham_PSD': sham_PSD}
with open(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis' + '\stroop_PSD.pkl', 'wb') as file:
    pickle.dump(msdict, file)

#% subject x session x fi = power
SR = 250
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
        fi_power = np.median(np.mean(tmp2 , axis=1), axis=0)
        ki = np.where(total_keylist==msid)[0][0]
        subject_x_se_x_fi_power[ki, t, :] = fi_power
        
subject_x_se_x_fi_power_beta = np.mean(subject_x_se_x_fi_power[:,:,17:27], axis=2)

#% subject x session = tscore
import stroop_score_calgen
import msFunction

pload = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db2.pkl'
with open(pload, 'rb') as file:
    msdict = pickle.load(file)
msdict_array = np.array(msdict)

tscore_save = []
for i in range(len(msdict_array)):
    data = np.array(msdict_array[i][[3,6,9,2,5,8]], dtype=float)  
    score, _ = stroop_score_calgen.stroop_tscore(data, weights = [[0.3, 0.7], [0.25, 0.25, 0.5]])
    tscore_save.append(score)
tscore_save = np.array(tscore_save)

total_keylist_groupinfo = []
subject_x_se_x_tscore = np.zeros((len(total_keylist), 3)) * np.nan
for ix, msid in enumerate(total_keylist):
    for t in range(len(sessions)):
        c0 = msdict_array[:,14] == msid
        c1 = np.logical_or((msdict_array[:,13] == sessions[t][0]), (msdict_array[:,13] == sessions[t][1]))
        conditions = [c0, c1]
        vix = np.where(np.logical_and.reduce(conditions))[0]  
        subject_x_se_x_tscore[ix, t] = np.mean(tscore_save[vix])
        
    ix = np.where(Z_group[:,3] == msid)[0][0]
    total_keylist_groupinfo.append(Z_group[ix][-1])
total_keylist_groupinfo = np.array(total_keylist_groupinfo)    

g0 = total_keylist_groupinfo=='VNS'
g1 = total_keylist_groupinfo=='sham'


valid_ix = np.logical_or(total_keylist_groupinfo=='VNS',\
                         total_keylist_groupinfo=='sham')
    
# valid_ix = total_keylist_groupinfo=='VNS'

subject_x_se_x_fi_power_beta2 = subject_x_se_x_fi_power_beta[valid_ix]
subject_x_se_x_tscore2 = subject_x_se_x_tscore[valid_ix]


from scipy.stats import pearsonr
delta_tscore1 = subject_x_se_x_tscore2[:,2] - subject_x_se_x_tscore2[:,0]
delta__beta1 = 10 * np.log10(subject_x_se_x_fi_power_beta2[:,2] / \
                             subject_x_se_x_fi_power_beta2[:,0])
# db?

delta_tscore2 = subject_x_se_x_tscore2[:,1] - subject_x_se_x_tscore2[:,0]
delta__beta2 = 10 * np.log10(subject_x_se_x_fi_power_beta2[:,1] / \
                             subject_x_se_x_fi_power_beta2[:,0])


delta_tscore = np.concatenate((delta_tscore1, delta_tscore2), axis=0)
delta__beta = np.concatenate((delta__beta1, delta__beta2), axis=0)

## Figure3 F

r_value, p_value = pearsonr(delta__beta, delta_tscore)
print(f"Pearson correlation coefficient: {r_value}")
print(f"P-value: {p_value}")

# 산점도 그리기

# 추세선 계산
coefficients = np.polyfit(delta__beta, delta_tscore, 1)
polynomial = np.poly1d(coefficients)
trendline = polynomial(delta__beta)

# 추세선 그리기
# plt.plot(delta__beta, trendline, color='lightcoral', linewidth=2, linestyle='-', label='Trendline')

# print('delta__beta', np.mean(delta__beta))
# print('delta_tscore', np.mean(delta_tscore))

# plt.scatter(delta__beta, delta_tscore)
# plt.xlabel('delta__beta')
# plt.ylabel('delta_tscore')
# plt.title(f'Scatter Plot and Pearson r: {r_value:.2f}')
# plt.grid(True)
# plt.show()

#%% bar graph, session x group 
# figure3 E

beta_bar_sham, beta_bar_VNS = [], []
for se in range(3):
    beta_bar_sham.append(np.mean(sham_PSD[se][:,17:27], axis=1))
    beta_bar_VNS.append(np.mean(vns_PSD[se][:,17:27], axis=1))

sham_means = [np.mean(session) for session in beta_bar_sham]
vns_means = [np.mean(session) for session in beta_bar_VNS]

sham_sems = [np.std(session, ddof=1) / np.sqrt(len(session)) for session in beta_bar_sham]
vns_sems = [np.std(session, ddof=1) / np.sqrt(len(session)) for session in beta_bar_VNS]

for i in range(3):
    t_stat, p_value = stats.ttest_ind(beta_bar_sham[i], beta_bar_VNS[i])
    print(i)
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

# # 바 그래프 그리기
# labels = ['Session 1', 'Session 2', 'Session 3']
# x = np.arange(len(labels))  # x축 위치

# ## Figure3 E

# fig, ax = plt.subplots()
# width = 0.35  # 막대 너비

# rects1 = ax.bar(x - width/2, sham_means, width, yerr=sham_sems, label='Sham', color='lightcoral', capsize=5)
# rects2 = ax.bar(x + width/2, vns_means, width, yerr=vns_sems, label='VNS', color='skyblue', capsize=5)

# # 그래프 설정
# ax.set_xlabel('Sessions')
# ax.set_ylabel('Values')
# ax.set_title('Comparison of Sham and VNS across Sessions')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.set_ylim(0.008, 0.015)
# ax.legend()

# # 그래프 보여주기
# plt.show()


#%% Figure3D

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
    
    
    # print(power, t)
    print(t, np.mean(fi_power_vns[:,17:27]), np.mean(fi_power_sham[:,17:27]))
    
    plt.plot(np.mean(fi_power_vns, axis=0)[4:], c=colors1[t], label='VNS_' + str(t))
    plt.plot(np.mean(fi_power_sham, axis=0)[4:], c=colors2[t], label='sham_' + str(t))
    # plt.title(str(t)) 
    plt.legend()

#%%
import pickle

# 필요한 변수를 딕셔너리에 저장
msdict_figure = {
    'target_freqs_05': target_freqs_05,
    'psd_pre_median_sham': psd_pre_median_sham,
    'psd_pre_sem_sham': psd_pre_sem_sham,
    'psd_post_median_sham': psd_post_median_sham,
    'psd_post_sem_sham': psd_post_sem_sham,
    'psd_pre_median_VNS': psd_pre_median_VNS,
    'psd_pre_sem_VNS': psd_pre_sem_VNS,
    'psd_post_median_VNS': psd_post_median_VNS,
    'psd_post_sem_VNS': psd_post_sem_VNS,
    'median_psd_3b_sham': median_psd_3b_sham,
    'sem_psd_3b_sham': sem_psd_3b_sham,
    'median_psd_3b_vns': median_psd_3b_vns,
    'sem_psd_3b_vns': sem_psd_3b_vns,
    'g1_means': g1_means,
    'g1_sems': g1_sems,
    'g0_means': g0_means,
    'g0_sems': g0_sems,
    'bands': bands,
    'figure3D_plot_data': figure3D_plot_data,
    'sham_means': sham_means,
    'sham_sems': sham_sems,
    'vns_means': vns_means,
    'vns_sems': vns_sems,
    'delta__beta': delta__beta,
    'delta_tscore': delta_tscore,
    'trendline': trendline
}

# 저장 경로 설정
spath = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure' + '\\'

# pickle 파일로 저장
with open(spath + 'vnstroop_figure3.pickle', 'wb') as file:
    pickle.dump(msdict_figure, file)

print("변수들이 성공적으로 저장되었습니다.")


#%%

def set_custom_yticks(ylim, num_divisions=6, roundn=3, offset=0):
    """ylim을 6분할하여 5개의 tick을 설정하는 함수"""
    ymin, ymax = ylim
    tick_interval = (ymax - ymin) / num_divisions
    yticks = np.round(np.arange(ymin, ymax * 1.001, tick_interval)[1:-1] + offset, roundn)
    return yticks

fig, axs = plt.subplots(2, 3, figsize=(4.36/2 * 3 * 1.2, 2.45/2 * 2 * 1.4))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# 1,1 자리에 해당하는 subplot에 plot하기 (Python 인덱스 기준으로 0,0 위치)

## A

alpha = 0.2
axs[0, 0].plot(target_freqs_05, psd_pre_median_sham, label='First Interval', c='#73E0D1', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_pre_median_sham - \
                       psd_pre_sem_sham, psd_pre_median_sham + psd_pre_sem_sham, alpha=alpha, color='#73E0D1')

axs[0, 0].plot(target_freqs_05, psd_post_median_sham, label='Second Interval', c='#49C4B1', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_post_median_sham - \
                       psd_post_sem_sham, psd_post_median_sham + psd_post_sem_sham, alpha=alpha, color='#49C4B1')

axs[0, 0].plot(target_freqs_05, psd_pre_median_VNS, label='First Interval_vns', c='#C7B6FF', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_pre_median_VNS - \
                       psd_pre_sem_VNS, psd_pre_median_VNS + psd_pre_sem_VNS, alpha=alpha, color='#C7B6FF')

axs[0, 0].plot(target_freqs_05, psd_post_median_VNS, label='Second Interval_vns', c='#A182FF', lw=1)
axs[0, 0].fill_between(target_freqs_05, psd_post_median_VNS - \
                       psd_post_sem_VNS, psd_post_median_VNS + psd_post_sem_VNS, alpha=alpha, color='#A182FF')

axs[0, 0].set_ylim(0.01, 0.045)
axs[0, 0].set_yticks(set_custom_yticks((0.01, 0.045), num_divisions=5))
axs[0, 0].set_xlim(0, 40)
axs[0, 0].set_xticks(np.arange(0, 40 + 1, 10))
axs[0, 0].set_ylabel('Relative Power', fontsize=7, labelpad=0.1)
axs[0, 0].set_xlabel('Frequency (Hz)', fontsize=7, labelpad=0.5)

## B

axs[0, 1].plot(target_freqs_05, median_psd_3b_sham, label='Sham', c='#73E0D1', lw=1)
axs[0, 1].fill_between(target_freqs_05, median_psd_3b_sham - sem_psd_3b_sham, \
                       median_psd_3b_sham + sem_psd_3b_sham, alpha=alpha, color='#73E0D1')

axs[0, 1].plot(target_freqs_05, median_psd_3b_vns, label='VNS', c='#C7B6FF', lw=1)
axs[0, 1].fill_between(target_freqs_05, median_psd_3b_vns - sem_psd_3b_vns, \
                       median_psd_3b_vns + sem_psd_3b_vns, alpha=alpha, color='#C7B6FF')
    
axs[0, 1].set_ylim(-3.5, 2.5)
axs[0, 1].set_yticks(set_custom_yticks((-4, 2.5), num_divisions=5, offset=0.1))
axs[0, 1].set_xlim(0, 40)
axs[0, 1].set_xticks(np.arange(0, 40 + 1, 10))
axs[0, 1].set_ylabel('Relative Power Difference', fontsize=7, labelpad=0.1)
axs[0, 1].set_xlabel('Frequency (Hz)', fontsize=7, labelpad=0.5)

## C

x = np.arange(4)
width = 0.4
axs[0, 2].bar(x - width/2, g1_means, width, yerr=g1_sems, capsize=2, label='sham', color='#B7FFF4', error_kw=dict(lw=0.6, capthick=0.6))
axs[0, 2].bar(x + width/2, g0_means, width, yerr=g0_sems, capsize=2, label='VNS', color='#E2DCFF', error_kw=dict(lw=0.6, capthick=0.6))

axs[0, 2].set_ylim(-3, 1.7)
axs[0, 2].set_yticks(set_custom_yticks((-3, 1.5), num_divisions=5, offset=0.3))
axs[0, 2].set_xticks(x)
axs[0, 2].set_xticklabels(bands)
axs[0, 2].set_ylabel('Mean of Power Difference', fontsize=7, labelpad=0.1)

## D
# Sham (Original): 밝은 청록색 (#B7FFF4)
# VNS (Original): 밝은 연보라색 (#E2DCFF)
# Sham (New): 진한 청록색 (#73E0D1)
# VNS (New): 진한 연보라색 (#C7B6FF)
# Sham (Extended): 더욱 진한 청록색 (#49C4B1)
# VNS (Extended): 더욱 진한 보라색 (#A182FF)
                           
colors1 = ['#B7FFF4', '#73E0D1', '#49C4B1']
colors2 = ['#E2DCFF', '#C7B6FF', '#A182FF']
alpha = 0.2
xaxis = np.arange(4, 40, dtype=int)
fi_ix = slice(4, 40)
for t in range(3):
    mean_sham = figure3D_plot_data[t][0]
    sem_sham = figure3D_plot_data[t][2]
    axs[1, 0].plot(xaxis, mean_sham[fi_ix], c=colors1[t], label='sham_' + str(t), lw=1)
    axs[1, 0].fill_between(xaxis, mean_sham[fi_ix] - sem_sham[fi_ix], mean_sham[fi_ix] + sem_sham[fi_ix], color=colors1[t], alpha=alpha)
    
    mean_vns = figure3D_plot_data[t][1]
    sem_vns = figure3D_plot_data[t][3]
    axs[1, 0].plot(xaxis, mean_vns[fi_ix], c=colors2[t], label='VNS_' + str(t), lw=1)
    axs[1, 0].fill_between(xaxis, mean_vns[fi_ix] - sem_vns[fi_ix], mean_vns[fi_ix] + sem_vns[fi_ix], color=colors2[t], alpha=alpha)

axs[1, 0].set_ylim(0.0, 0.027)
axs[1, 0].set_yticks(set_custom_yticks((0.0, 0.027), num_divisions=5, offset=0))
axs[1, 0].set_xlim(0, 44)
axs[1, 0].set_xticks(np.arange(0, 40 + 1, 10))
axs[1, 0].set_ylabel('Relative Power', fontsize=7, labelpad=0.1)
axs[1, 0].set_xlabel('Frequency (Hz)', fontsize=7, labelpad=0.5)
## D

# 바 그래프 그리기
labels = ['Session 1', 'Session 2', 'Session 3']
x = np.arange(len(labels))  # x축 위치
width = 0.35  # 막대 너비

axs[1, 1].bar(x - width/2, sham_means, width, yerr=sham_sems, label='Sham', color='#B7FFF4', capsize=2, error_kw=dict(lw=0.6, capthick=0.6))
axs[1, 1].bar(x + width/2, vns_means, width, yerr=vns_sems, label='VNS', color='#E2DCFF', capsize=2, error_kw=dict(lw=0.6, capthick=0.6))

axs[1, 1].set_ylim(0.008, 0.018)
axs[1, 1].set_yticks(set_custom_yticks((0.008, 0.018), num_divisions=5, offset=0))
# axs[1, 0].set_xlim(4, len(mean_vns))
axs[1, 1].set_ylabel('Mean of Relative Power', fontsize=7, labelpad=0.1)
axs[1, 1].set_xticks(x)
axs[1, 1].set_xlabel('Sessions #', fontsize=7, labelpad=0.5)
axs[1, 1].set_xticklabels(range(1,4))

## F
axs[1, 2].plot(delta__beta, trendline, color='lightcoral', linewidth=1, linestyle='-', label='Trendline', alpha=0.7)
axs[1, 2].scatter(delta__beta, delta_tscore, s=7, c='#A182FF', alpha=0.7, edgecolors='none')

axs[1, 2].set_ylim(-0.2, 0.9)
axs[1, 2].set_yticks(set_custom_yticks((-0.2, 0.9), num_divisions=5, offset=-0.02))
axs[1, 2].set_ylabel('Δ Stroop Score', fontsize=7, labelpad=0.1)

axs[1, 2].set_xlim(-0.35, 0.55)
axs[1, 2].set_xticks(np.arange(-0.3, 0.5 * 1.00001, 0.2))
axs[1, 2].set_xlabel('Δ Beta Power', fontsize=7, labelpad=0.5)


###    
for ax in axs.flat:
    ax.tick_params(axis='x', labelsize=7, pad=2, width=0.2)  # x tick 폰트 크기 7로 설정
    ax.tick_params(axis='y', labelsize=7, pad=0.2, width=0.2)  # y tick 폰트 크기 7로 설정
    
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure3.png'
    # plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()
#%%
# axs[0, 0].set_xlabel('Frequency (Hz)')
# axs[0, 0].set_ylabel('Relative Power Spectral Density (PSD)')
# axs[0, 0].legend()
# axs[0, 0].set_title('PSD with Median and SEM')



#%%

if False:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure3.png'
    # plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

#B7FFF4,  #C2E6E1
#E2DCFF, #CFD2E9



# 플롯 설정
plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Power Change (dB)')
# plt.legend()
plt.title('PSD with Median and SEM')

if False:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure3B.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()
    









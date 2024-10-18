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

#%%

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

#%% Figure3A
fig1 = plt.figure()
psd_pre_save, psd_post_save = np.array(psd_pre_save), np.array(psd_post_save)

label_info3 = label_info2[:len(psd_pre_save)]
g0 = np.where(label_info3=='VNS')[0]
g1 = np.where(label_info3=='sham')[0]

s_fi, e_if = 1, 40
target_freqs_05 = np.linspace(s_fi, e_if, psd_pre_save.shape[1])

alpha = 0.5

median_psd = np.median(psd_pre_save[g0], axis=0)
sem_psd = sem(psd_pre_save[g0], axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval_vns', c='#E2DCFF')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=alpha, color='#E2DCFF')

median_psd_2 = np.median(psd_post_save[g0], axis=0)
sem_psd_2 = sem(psd_post_save[g0], axis=0)
plt.plot(target_freqs_05, median_psd_2, label='Second Interval_vns', c='#CFD2E9')
plt.fill_between(target_freqs_05, median_psd_2 - sem_psd_2, median_psd_2 + sem_psd_2, alpha=alpha, color='#CFD2E9')

median_psd = np.median(psd_pre_save[g1], axis=0)
sem_psd = sem(psd_pre_save[g1], axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval', c='#B7FFF4')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=alpha, color='#B7FFF4')

median_psd_2 = np.median(psd_post_save[g1], axis=0)
sem_psd_2 = sem(psd_post_save[g1], axis=0)
plt.plot(target_freqs_05, median_psd_2, label='Second Interval', c='#C2E6E1')
plt.fill_between(target_freqs_05, median_psd_2 - sem_psd_2, median_psd_2 + sem_psd_2, alpha=alpha, color='#C2E6E1')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Relative Power Spectral Density (PSD)')
plt.legend()
plt.title('PSD with Median and SEM')
# plt.show()

if False:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure3A.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

#B7FFF4,  #C2E6E1
#E2DCFF, #CFD2E9


#%% Figure3B
fig2 = plt.figure()
alpha = 0.7
g0data = 10*np.log10(psd_post_save[g0]/psd_pre_save[g0])

median_psd = np.median(g0data, axis=0)
sem_psd = sem(g0data, axis=0)
plt.plot(target_freqs_05, median_psd, label='VNS', c='#CFD2E9')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=alpha, color='#CFD2E9')

g1data = 10*np.log10(psd_post_save[g1]/psd_pre_save[g1])

median_psd_2 = np.median(g1data, axis=0)
sem_psd_2 = sem(g1data, axis=0)
plt.plot(target_freqs_05, median_psd_2, label='Sham', c='#C2E6E1')
plt.fill_between(target_freqs_05, median_psd_2 - sem_psd_2, median_psd_2 + sem_psd_2, alpha=alpha, color='#C2E6E1')

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
    
#%%


#%% Figure3C

import numpy as np
import matplotlib.pyplot as plt

# Frequency interval indices for different bands
theta_band = slice(4, 8)    # Example: 0-7 index for theta band (4-8 Hz)
alpha_band = slice(8, 15)   # Example: 8-14 index for alpha band (8-13 Hz)
beta_band = slice(15, 30)   # Example: 15-29 index for beta band (13-30 Hz)
gamma_band = slice(30, 39)  # Example: 30-38 index for gamma band (30-40 Hz)

# Compute mean for each frequency band
theta_g0_mean = np.mean(g0data[:, theta_band], axis=1)
alpha_g0_mean = np.mean(g0data[:, alpha_band], axis=1)
beta_g0_mean = np.mean(g0data[:, beta_band], axis=1)
gamma_g0_mean = np.mean(g0data[:, gamma_band], axis=1)

theta_g1_mean = np.mean(g1data[:, theta_band], axis=1)
alpha_g1_mean = np.mean(g1data[:, alpha_band], axis=1)
beta_g1_mean = np.mean(g1data[:, beta_band], axis=1)
gamma_g1_mean = np.mean(g1data[:, gamma_band], axis=1)

# Prepare data for bar plot
bands = ['Theta', 'Alpha', 'Beta', 'Gamma']
g0_means = [np.mean(theta_g0_mean), np.mean(alpha_g0_mean), np.mean(beta_g0_mean), np.mean(gamma_g0_mean)]
g1_means = [np.mean(theta_g1_mean), np.mean(alpha_g1_mean), np.mean(beta_g1_mean), np.mean(gamma_g1_mean)]

def compute_sem(data, band):
    return np.std(data[:, band], axis=1) / np.sqrt(data.shape[0])

theta_g0_sem = compute_sem(g0data, theta_band)
alpha_g0_sem = compute_sem(g0data, alpha_band)
beta_g0_sem = compute_sem(g0data, beta_band)
gamma_g0_sem = compute_sem(g0data, gamma_band)

theta_g1_sem = compute_sem(g1data, theta_band)
alpha_g1_sem = compute_sem(g1data, alpha_band)
beta_g1_sem = compute_sem(g1data, beta_band)
gamma_g1_sem = compute_sem(g1data, gamma_band)

# Prepare SEM data for bar plot
g0_sems = [np.mean(theta_g0_sem), np.mean(alpha_g0_sem), np.mean(beta_g0_sem), np.mean(gamma_g0_sem)]
g1_sems = [np.mean(theta_g1_sem), np.mean(alpha_g1_sem), np.mean(beta_g1_sem), np.mean(gamma_g1_sem)]

# Plotting
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, g1_means, width, yerr=g1_sems, capsize=5, label='sham', color='#B7FFF4')
bars2 = ax.bar(x + width/2, g0_means, width, yerr=g0_sems, capsize=5, label='VNS', color='#E2DCFF')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean dB')
ax.set_title('Mean Power by Frequency Band')
ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.legend()

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure3C.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

















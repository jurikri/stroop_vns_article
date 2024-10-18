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
import VNS_PPG_def

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


#%%

beta1, beta2 = 1, 1
tix = eeg_df[:,-1]
X = np.array(eeg_df[:,[2]])
        
#%%

id_save, psd_pre_save, psd_post_save =  [], [], []
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
    X2_pre = np.transpose(np.array(X[vix2]))
    

    s = last_cue + (0.5*60)
    e = last_cue + (5.5*60)
    vix = np.where(np.logical_and((tix > s), (tix < e)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    X2_post = np.transpose(np.array(X[vix2]))

    id_save.append(id_i)
    
    SDNN, RMSSD, pNN50, BPM, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=np.array(X2_pre)[0])
    psd_pre_save.append([SDNN, RMSSD, pNN50, BPM])
    
    SDNN, RMSSD, pNN50, BPM, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=np.array(X2_post)[0])
    psd_post_save.append([SDNN, RMSSD, pNN50, BPM])

#%%

psd_pre_save, psd_post_save = np.array(psd_pre_save), np.array(psd_post_save)

label_info3 = label_info2
g0 = np.where(label_info3=='VNS')[0]
g1 = np.where(label_info3=='sham')[0]

print('VNS')
target = psd_pre_save[g0]
vix = np.where(target[:,2] < 10)[0]
print(np.mean(target[vix], axis=0))

target = psd_post_save[g0]
vix = np.where(target[:,2] < 10)[0]
print(np.mean(target[vix], axis=0))

print('SHAM')
target = psd_pre_save[g1]
vix = np.where(target[:,2] < 10)[0]
print(np.mean(target[vix], axis=0))

target = psd_post_save[g1]
vix = np.where(target[:,2] < 10)[0]
print(np.mean(target[vix], axis=0))






#%% stroop session 기준

id_save, sessions =  [], [[],[],[]]
idlist = list(experiment_data.keys())
# id_i = 0
for id_i in tqdm(range(len(idlist))):
    msid = idlist[id_i]
    

    trial = 0
    directory_path = experiment_data[msid][trial]
    pkl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

    pkl_files.sort()
    with open(pkl_files[0], 'rb') as file:
        msdict = pickle.load(file)
        session1_start = msdict['cue_time_stamp']
        
        #
    trial = 1
    directory_path = experiment_data[msid][trial]
    pkl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

    pkl_files.sort()
    with open(pkl_files[-1], 'rb') as file:
        msdict = pickle.load(file)
        session1_end = msdict['cue_time_stamp']
    #
    trial = 2
    directory_path = experiment_data[msid][trial]
    pkl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

    pkl_files.sort()
    with open(pkl_files[0], 'rb') as file:
        msdict = pickle.load(file)
        session2_start = msdict['cue_time_stamp']
        
        #
    trial = 3
    directory_path = experiment_data[msid][trial]
    pkl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

    pkl_files.sort()
    with open(pkl_files[-1], 'rb') as file:
        msdict = pickle.load(file)
        session2_end = msdict['cue_time_stamp']
        
    #
    trial = 4
    directory_path = experiment_data[msid][trial]
    pkl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

    pkl_files.sort()
    with open(pkl_files[0], 'rb') as file:
        msdict = pickle.load(file)
        session3_start = msdict['cue_time_stamp']
        
        #
    trial = 5
    directory_path = experiment_data[msid][trial]
    pkl_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))

    pkl_files.sort()
    with open(pkl_files[-1], 'rb') as file:
        msdict = pickle.load(file)
        session3_end = msdict['cue_time_stamp']
             
    times = np.array([session1_start, session1_end, session2_start, session2_end, session3_start, session3_end])
    times2 = times - session1_start
    print(msid, times2)

    vix = np.where(np.logical_and((tix > session1_start), (tix < session1_end)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    SDNN, RMSSD, pNN50, BPM, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=np.array(X[vix2])[:,0])
    sessions[0].append([SDNN, RMSSD, pNN50, BPM])
    
    vix = np.where(np.logical_and((tix > session2_start), (tix < session2_end)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    SDNN, RMSSD, pNN50, BPM, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=np.array(X[vix2])[:,0])
    sessions[1].append([SDNN, RMSSD, pNN50, BPM])
    
    vix = np.where(np.logical_and((tix > session3_start), (tix < session3_end)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    SDNN, RMSSD, pNN50, BPM, peaks = VNS_PPG_def.msmain(SR=SR, ppg_data=np.array(X[vix2])[:,0])
    sessions[2].append([SDNN, RMSSD, pNN50, BPM])



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

#%%
sessions[0], sessions[1], sessions[2] = np.array(sessions[0]), np.array(sessions[1]), np.array(sessions[2])

label_info3 = label_info2
g0 = np.where(label_info3=='VNS')[0]
g1 = np.where(label_info3=='sham')[0]

print()
print('VNS')
print(np.median(sessions[1][g0] - sessions[0][g0], axis=0))
print(np.median(sessions[2][g0] - sessions[0][g0], axis=0))

print()
print('sham')
print(np.median(sessions[1][g1] - sessions[0][g1], axis=0))
print(np.median(sessions[2][g1] - sessions[0][g1], axis=0))



#%%

s_fi, e_if = 1, 40
target_freqs_05 = np.linspace(s_fi, e_if, psd_pre_save.shape[1])

median_psd = np.median(psd_pre_save[g0], axis=0)
sem_psd = sem(psd_pre_save[g0], axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval_vns')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)

median_psd_2 = np.median(psd_post_save[g0], axis=0)
sem_psd_2 = sem(psd_post_save[g0], axis=0)
plt.plot(target_freqs_05, median_psd_2, label='Second Interval_vns')
plt.fill_between(target_freqs_05, median_psd_2 - sem_psd_2, median_psd_2 + sem_psd_2, alpha=0.3)

median_psd = np.median(psd_pre_save[g1], axis=0)
sem_psd = sem(psd_pre_save[g0], axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)

median_psd_2 = np.median(psd_post_save[g1], axis=0)
sem_psd_2 = sem(psd_post_save[g0], axis=0)
plt.plot(target_freqs_05, median_psd_2, label='Second Interval')
plt.fill_between(target_freqs_05, median_psd_2 - sem_psd_2, median_psd_2 + sem_psd_2, alpha=0.3)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.legend()
plt.title('PSD with Median and SEM')
plt.show()
#%%

g0data = 10*np.log10(psd_post_save[g0]/psd_pre_save[g0])

median_psd = np.mean(g0data, axis=0)
sem_psd = sem(g0data, axis=0)
plt.plot(target_freqs_05, median_psd, label='VNS')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)

g1data = 10*np.log10(psd_post_save[g1]/psd_pre_save[g1])

median_psd_2 = np.mean(g1data, axis=0)
sem_psd_2 = sem(g1data, axis=0)
plt.plot(target_freqs_05, median_psd_2, label='Sham')
plt.fill_between(target_freqs_05, median_psd_2 - sem_psd_2, median_psd_2 + sem_psd_2, alpha=0.3)

# 플롯 설정
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.legend()
plt.title('PSD with Median and SEM')
plt.show()
























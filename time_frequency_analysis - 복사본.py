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





















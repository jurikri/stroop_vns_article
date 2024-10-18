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
import msanalysis
import sys;

import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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

from scipy import stats

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
                    file_info['msid'], file_info['trial'], file_info['inter_trial'], msdict['Correct']]
        
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
# X = np.transpose(X, (0, 2, 1))


# 이 버전의 SOSO 기기는 +-40940에 달했을때 데이터가 소실된다.
# 따라서 해당데이터는 제외함.
vix, cnt = [], 0
for i in range(X.shape[0]):
    if not(np.sum(X[i] >= 40940) > 0 or np.sum(X[i] <= -40940)):
        vix.append(i)
    else:
        cnt += 1
        
X, Z = X[vix], Z[vix]
#%% nmr
# 0 ~ 150 ms의 dev를 사용하여 scale을 맞추고
# 150 ms가 통일된 offset이 되도록 함

SR = 250
# standard_signal = np.median(X, axis=(0,2))
# standard_signal[:int(SR*(0.25+0.5))]


# offset_standard = np.mean(standard_signal[int(SR*(0.20)):int(SR*(0.20+0.15))])
# std_stadard = np.std(standard_signal[:int(SR*(0.20+0.15))+1] - offset_standard)

# -200 ~ 250 ms의 dev를 사용하여 scale을 맞추고
# 250 ms가 통일된 offset이 되도록 함

# offset_standard = np.(standard_signal[:int(SR*(0.20+0.25))])
# std_stadard = np.std(standard_signal[:int(SR*(0.20+0.25))+1] - offset_standard)
# scale, offset

# Xnmr = np.zeros(X.shape) * np.nan
# for ch in range(2):
#     Xch = X[:,:,ch]
    
#     # plt.plot(np.median(Xch[:,:int(SR*(0.20+0.25))], axis=1))
#     stds = np.std(Xch[:, int(SR*(0.20)):int(SR*(0.20+0.1))], axis=1)
#     stds_expanded = stds.reshape(-1, 1)
#     tmp1 = (Xch / stds_expanded)
    
    
#     offsets = np.median((tmp1[:, int(SR*(0.20)):int(SR*(0.20+0.1))]), axis=1)
#     offsets_expanded = offsets.reshape(-1, 1)

#     tmp2 = tmp1 - offsets_expanded
#     # plt.plot(np.median(Xch_offsets[:,:int(SR*(0.20+0.25))], axis=1))

   
    
#     Xnmr[:,:,ch] = tmp2
    
    # Xnmr[:,:,ch]  = Xnmr[:,:,ch] - Xnmr[:,int(SR*(0.20+0.25)),ch].reshape(-1, 1)
    
# np.where(np.mean(Xnmr, axis=(1,2)) > 1000)[0]
# plt.plot(X[2969,:,0])

# plt.plot(np.median(Xnmr, axis=(0,2)))

#%%
print(Z[0])
vix = np.where(Z[:,1]=='맞음')[0]
plt.plot(np.median(X[vix], axis=(0,2)))

#%%

#%#
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

#% # 개별 데이터 기준, 맞는것만, trial 순서로, cue에 대해, group 비교
X_chmean = np.mean(X, axis=2)
# plt.plot(np.median(X_chmean, axis=(0)))

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

#%% Figure2 se mean

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

xaxis = np.linspace(-200, ((420/SR)-0.2)*1000 , 420)
colors1 = plt.cm.Blues(np.linspace(0.3, 1, 6))  # Blue shades for group 1
colors2 = plt.cm.Reds(np.linspace(0.3, 1, 6))   # Red shades for group 2

# cue = 'incongruent'

cues = ['neutral', 'congruent', 'incongruent']
session = [[2,3,4,5]]

barplot = {'neutral':[[],[]], \
            'congruent':[[],[]], \
            'incongruent':[[],[]]}
    
# barplot = {'neutral':{'True':[[],[]], 'False':[[],[]]}, \
#            'congruent':{'True':[[],[]], 'False':[[],[]]}, \
#            'incongruent':{'True':[[],[]], 'False':[[],[]]}}

# ERPsave = {}
ans = '맞음'
for cue in cues:
    plt.figure(figsize=(6, 4))
    c1 = (Z_group[:,0] == cue)
    # for ans in ['맞음', '틀림']:
    for t in range(6):
        if ans == '맞음': anslabel = 'Correct'
        if ans == '틀림': anslabel = 'Incorrect'
        
        # c2 = (Z_group[:,1] == ans)
        c2 = (Z_group[:,1] == ans)
        
        g_vns = (Z_group[:,7] == 'VNS')
        g_sham = (Z_group[:,7] == 'sham')
        
        vns_plots = []
        sham_plots = []
        
        SR = 250
        s,e = int((0)*SR), int((0.20+1.5)*SR)
        
        yvalues_vnss, yvalues_shams = [], []
        c0 = (np.array(Z_group[:,4], dtype=float) == t)
        # c0 = np.logical_and((np.array(Z_group[:,4], dtype=float) < 4), (np.array(Z_group[:,4], dtype=float) > 1))
        
        # 예제 조건 및 데이터
        conditions = [c0, c1, g_vns]
        vix = np.where(np.logical_and.reduce(conditions))[0]
    
        yvalues_vns = np.median(np.array(X_chmean)[vix], axis=0)
        sem_vns = sem(np.array(X_chmean)[vix], axis=0, nan_policy='omit')
        yvalues_smoothing_vns = moving_average_smoothing(yvalues_vns, 20)
        
        barplot[cue][0].append(np.array(X_chmean)[vix])
        
        plt.plot(xaxis[s:e], yvalues_smoothing_vns[s:e], \
                  label=cue + '_VNS_' + str(anslabel), linewidth=2, c=colors1[t])
        # plt.fill_between(xaxis[s:e], 
        #                   (yvalues_smoothing_vns - sem_vns)[s:e], 
        #                   (yvalues_smoothing_vns + sem_vns)[s:e], 
        #                   color='blue', alpha=0.3)
        
        conditions = [c0, c1, g_sham]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        yvalues_sham = np.median(np.array(X_chmean)[vix], axis=0)
        sem_sham = sem(np.array(X_chmean)[vix], axis=0, nan_policy='omit')
        yvalues_smoothing_sham = moving_average_smoothing(yvalues_sham, 20)
        
        barplot[cue][1].append(np.array(X_chmean)[vix])
        
        plt.plot(xaxis[s:e], yvalues_smoothing_sham[s:e], \
                  label=cue + '_SHAM_' + str(anslabel), linewidth=2, c=colors2[t])
        # plt.fill_between(xaxis[s:e], 
        #                   (yvalues_smoothing_sham - sem_sham)[s:e], 
        #                   (yvalues_smoothing_sham + sem_sham)[s:e], 
        #                   color='red', alpha=0.3)
        
        #%%
        
        # ERPsave[cue] = np.array([yvalues_vnss, yvalues_shams])
        plt.axvspan(xaxis[s:e][int((0.2)*SR)], xaxis[s:e][int((0.35)*SR)], color='gray', alpha=0.1)
        plots = sham_plots + vns_plots
        labels = [plot.get_label() for plot in plots]
        # plt.legend()
        
    if True: # figure save codes
        # plt.legend(plots, labels)
        plt.xlabel('Time (ms) from Cue')
        plt.ylabel('Relative Voltage')
        plt.title(cue)
        fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
        fsave2 = fsave + '\\' + cue + '_ERP.png'
        plt.tight_layout()
        plt.legend()
        plt.savefig(fsave2, dpi=100 ,bbox_inches='tight', pad_inches=0.1)
        plt.show()

#%% barplot
from scipy.stats import ttest_ind


s, e = int((0.2 + 0.2)*SR), int((0.2 + 0.35)*SR)
cues = ['neutral', 'congruent', 'incongruent']

for cue in cues:
    means_vns = []
    errors_vns = []
    means_sham = []
    errors_sham = []
    labels = []
    
    plt.figure()
    for t in range(6):
        if ans == '맞음': anslabel = 'Correct Response'
        if ans == '틀림': anslabel = 'Incorrect Response'
        
        data_vns = np.mean(barplot[cue][0][t][:, s:e], axis=1)
        data_sham = np.mean(barplot[cue][1][t][:, s:e], axis=1)
        
        # print(np.mean(data_vns))
        # print(np.mean(data_sham))
        t_stat, p_value = ttest_ind(data_vns, data_sham)
        print(f"{cue}_{anslabel}", p_value)


        mean_vns = np.mean(data_vns)
        error_vns = sem(data_vns, nan_policy='omit')
        mean_sham = np.mean(data_sham)
        error_sham = sem(data_sham, nan_policy='omit')

        means_vns.append(mean_vns)
        errors_vns.append(error_vns)
        means_sham.append(mean_sham)
        errors_sham.append(error_sham)
        labels.append(f"{cue}_{anslabel}")

    # 막대 그래프 그리기
    x_pos = np.arange(len(labels))
    bar_width = 0.35
    
    plt.bar(x_pos - bar_width/2, means_vns, bar_width, yerr=errors_vns, \
            align='center', alpha=0.7, ecolor='black', capsize=10, label='VNS')
        
    plt.bar(x_pos + bar_width/2, means_sham, bar_width, yerr=errors_sham, \
            align='center', alpha=0.7, ecolor='black', capsize=10, label='Sham')
    
    plt.xticks(x_pos, labels)
    plt.xlabel('Conditions')
    plt.ylabel('Mean Value with SEM')
    plt.title('Bar Graph with SEM for VNS and Sham Conditions')
    plt.legend()
    
    # y축 범위 설정
    min_y = min(min(means_vns) - max(errors_vns), min(means_sham) - max(errors_sham)) - 10 * max(errors_vns + errors_sham)
    max_y = max(max(means_vns) + max(errors_vns), max(means_sham) + max(errors_sham)) + 10 * max(errors_vns + errors_sham)
    plt.ylim(min_y, max_y)
    
    plt.tight_layout()
    plt.show()

#%%
cues = ['neutral', 'congruent', 'incongruent']
s, e = int((0.2 + 0.2)*SR), int((0.2 + 0.35)*SR)
for cue in cues:
    start_se = [0,1]
    end_se = [4,5]
    
    barplot[cue][0][start_se[0]][:,s:e]
    
    """
    사람 by 사람으로, start session N2b와 마지막 sessiion N2b가 다르지 않음을 보임
    
    """


#%%

"""
1. 위 코드 cue 3가지 자동화, figure save
2. 데이터 받아서 밑에 분석 진행
3. performance와 corr 분석


"""


#%% 정량화
SR = 250
s = int(SR*(0.25 + 0.3))
e = int(SR*(0.25 + 0.4))

cues = ['neutral', 'congruent', 'incongruent']
session = [[0,1], [2,3], [4,5]]
# ERPsave = {}
for cue in cues:
    
    yvalues_vnss, yvalues_shams = [], []
    for trial in range(6):
        c1 = (Z_group[:,0] == cue)
        c2 = (Z_group[:,1] == '맞음')
        c3 = (Z_group[:,4] == str(trial))
        
        g_vns = (Z_group[:,6] == 'VNS')
        g_sham = (Z_group[:,6] == 'sham')
        
        conditions = [c1, c2, c3, g_vns]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        n450_vns = np.mean(X_chmean[vix][:,s:e], axis=1)
        
        conditions = [c1, c2, c3, g_sham]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        n450_sham = np.mean(X_chmean[vix][:,s:e], axis=1)
        
        yvalues_vnss.append(n450_vns)
        yvalues_shams.append(n450_sham)
    
    # 평균과 표준오차(SEM) 계산
    # VNS
    tdata = list(yvalues_vnss)
    mean_data = [np.nanmedian(time_data) for time_data in tdata]
    sem_data = [np.std(time_data, ddof=1) / np.sqrt(len(time_data)) for time_data in tdata]
    plt.figure(figsize=(8, 5))
    xaxis = range(1, len(mean_data)+1)
    plt.errorbar(xaxis, mean_data, yerr=sem_data, fmt='o', markersize=10, capsize=5, color='b')
    plt.plot(xaxis, mean_data, linestyle='-', color='b', label='VNS')

    # Sham
    tdata = list(yvalues_shams)
    mean_data = [np.nanmedian(time_data) for time_data in tdata]
    sem_data = [np.std(time_data, ddof=1) / np.sqrt(len(time_data)) for time_data in tdata]
    plt.errorbar(xaxis, mean_data, yerr=sem_data, fmt='o', markersize=10, capsize=5, color='r')
    plt.plot(xaxis, mean_data, linestyle='-', color='r', label='Sham')
    # plt.title('Line Plot with Data Points and Error Bars')
    plt.xlabel('Trial')
    plt.ylabel('Relatvie Voltage 200-350 ms after cue')
    plt.grid(True)
    plt.title(cue)
    plt.legend()
    
    if False:
        fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
        fsave2 = fsave + '\\' + cue + '_ERP_N300.png'
        plt.tight_layout()
        plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
        plt.show()

#%% 정량화 수정
SR = 250
s = int(SR*(0.2 + 0.3))
e = int(SR*(0.2 + 0.4))

cues = ['neutral', 'congruent', 'incongruent']
session = [[0,1], [2,3], [4,5]]
for cue in cues:
    yvalues_vnss, yvalues_shams = [], []
    for t in range(len(session)):
        c1 = (Z_group[:,0] == cue)
        c2 = (Z_group[:,1] == '맞음')
        c3 = np.logical_or(Z_group[:,4] == str(session[t][0]), Z_group[:,4] == str(session[t][1]))
        
        g_vns = (Z_group[:,6] == 'VNS')
        g_sham = (Z_group[:,6] == 'sham')
        
        conditions = [c1, c2, g_vns]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        n450_vns = np.nanmedian(X_chmean[vix][:,s:e], axis=1)
        
        conditions = [c1, c2, g_sham]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        n450_sham = np.nanmedian(X_chmean[vix][:,s:e], axis=1)
        
        yvalues_vnss.append(n450_vns)
        yvalues_shams.append(n450_sham)
    
    # 평균과 표준오차(SEM) 계산
    # VNS
    tdata = list(yvalues_vnss)
    mean_data = [np.nanmedian(time_data) for time_data in tdata]
    sem_data = [np.std(time_data, ddof=1) / np.sqrt(len(time_data)) for time_data in tdata]
    plt.figure(figsize=(8, 5))
    xaxis = range(1, len(mean_data)+1)
    plt.errorbar(xaxis, mean_data, yerr=sem_data, fmt='o', markersize=10, capsize=5, color='b')
    plt.plot(xaxis, mean_data, linestyle='-', color='b', label='VNS')

    # Sham
    tdata = list(yvalues_shams)
    mean_data = [np.nanmedian(time_data) for time_data in tdata]
    sem_data = [np.std(time_data, ddof=1) / np.sqrt(len(time_data)) for time_data in tdata]
    plt.errorbar(xaxis, mean_data, yerr=sem_data, fmt='o', markersize=10, capsize=5, color='r')
    plt.plot(xaxis, mean_data, linestyle='-', color='r', label='Sham')
    # plt.title('Line Plot with Data Points and Error Bars')
    plt.xlabel('Trial')
    plt.ylabel('Relatvie Voltage 200-350 ms after cue')
    plt.grid(True)
    plt.title(cue)
    plt.legend()
    
    if False:
        fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
        fsave2 = fsave + '\\' + cue + '_ERP_N300.png'
        plt.tight_layout()
        plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
        plt.show()



    #%%
    
    print(trial, np.mean(n450_vns), np.mean(n450_sham))


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


#%%

import numpy as np
import matplotlib.pyplot as plt

# 예제 데이터 생성 (5 x 10 matrix)
np.random.seed(0)
data = np.random.rand(5, 10)

# 각 열의 평균 계산
mean_data = np.mean(data, axis=0)

# 표준오차(SEM) 계산
sem_data = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(mean_data)), mean_data, yerr=sem_data, fmt='o', markersize=10, capsize=5, color='b')
plt.plot(mean_data, linestyle='-', color='b')
plt.title('Line Plot with Data Points and Error Bars')
plt.xlabel('Time')
plt.ylabel('Mean Value')
plt.grid(True)
plt.show()



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

#%% ERP 추가분석

"""

"""

print(X.shape)
print('X_chmean.shape', X_chmean.shape)
print(Z_group.shape)
print(Z_group[0])



SR = 250
s, e = int(round(SR*(0.2+0.3))), int(round(SR*(0.2+0.4))) # 200~350 ms


c1 = (Z_group[:,7] == 'VNS')
ans = '맞음'

for ans in ['틀림']:
    for cue in ['incongruent']:
        c0 = (Z_group[:,1] == ans)
        c2 = (Z_group[:,0] == cue)
        c3 = (np.array(Z_group[:,4], dtype=float) > 1)
        conditions = [c0, c1, c2, c3]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        rt = np.array(Z_group[vix][:,2], dtype=float)
        
        print(len(vix))
        N2b = np.nanmin(X_chmean[vix][:,s:e], axis=1)
        
        # cc1 = np.logical_and(np.isnan(rt)==0, np.isnan(N2b)==0)
        # cc2 = np.logical_and(N2b < 100000, N2b > -100000)
        # andvix = np.logical_and(cc1, cc2)
        
        corr1, corr2 = rt, N2b
        
        corr, p_value = pearsonr(corr1, corr2)
        print(corr)
        plt.figure()
        plt.scatter(corr1, corr2, s=0.5)

 

# rt = np.array(Z_group[vix][:,2], dtype=float)

# corr1, corr2 = np.array(N2b), np.array(rt)

# from scipy.stats import pearsonr
# corr, p_value = pearsonr(corr1, corr2)
# print(corr)
#%% subject grouping
mskey_list = list(set(Z_group[:,3]))
ingroup = np.logical_or((Z_group[:,-1] == 'sham'), (Z_group[:,-1] == 'VNS'))
stimulus_types = np.array(['neutral'])
mssave = []
for i in range(len(mskey_list)):
    for j in range(len(stimulus_types)):
        stimulus = stimulus_types[j]
        msid = mskey_list[i]
        
        c0 = (Z_group[:,3] == msid)
        c1 = (Z_group[:,0] == stimulus)
        c2 = np.logical_or(Z_group[:,4] == str(4), \
                           Z_group[:,4] == str(5))
        
        conditions = [c0, c1, c2, ingroup]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        if len(vix) > 0:
            # mrt = np.nanmedian(np.array(Z_group[vix][:,2], dtype=float))
            
            acc = np.nanmean(Z_group[vix][:,1] == '맞음')
            rt = np.nanmean(np.array(Z_group[vix][:,2], dtype=float))
            
            mssave.append([msid, acc, rt, Z_group[vix[0],-1]])

mssave = np.array(mssave)

#%%
lown = 20
vix = np.where(mssave[:,-1]=='VNS')[0]
VNS_low = vix[np.argsort(np.array(mssave[vix,2], dtype=float))[::-1][:lown]]
vix = np.where(mssave[:,-1]=='sham')[0]
sham_low = vix[np.argsort(np.array(mssave[vix,2], dtype=float))[::-1][:lown]]

lowlist = list(mssave[VNS_low, 0]) + list(mssave[sham_low, 0])

#%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

xaxis = np.linspace(-200, ((420/SR)-0.2)*1000 , 420)
colors1 = plt.cm.Blues(np.linspace(0.3, 1, 4))  # Blue shades for group 1
colors2 = plt.cm.Reds(np.linspace(0.3, 1, 4))   # Red shades for group 2

# cue = 'incongruent'

cues = ['neutral', 'congruent', 'incongruent']

barplot = {'neutral':{'맞음':[[],[]], '틀림':[[],[]]}, \
            'congruent':{'맞음':[[],[]], '틀림':[[],[]]}, \
            'incongruent':{'맞음':[[],[]], '틀림':[[],[]]}}

ans = '맞음'

inlow = []
for i in range(len(Z_group)):
    if Z_group[i,3] in lowlist:
        inlow.append(1)
    else:
        inlow.append(0)
inlow = np.array(inlow)

for cue in cues:
    plt.figure(figsize=(6, 4))
    c1 = (Z_group[:,0] == cue)
    # for ans in ['맞음', '틀림']:
    if ans == '맞음': anslabel = 'Correct'
    if ans == '틀림': anslabel = 'Incorrect'
    
    # c2 = (Z_group[:,1] == ans)
    c2 = (Z_group[:,1] == ans)
    
    g_vns = (Z_group[:,7] == 'VNS')
    g_sham = (Z_group[:,7] == 'sham')
    
    Z_group[0]
    
    vns_plots = []
    sham_plots = []
    
    SR = 250
    s,e = int((0)*SR), int((0.20+1.5)*SR)
    
    yvalues_vnss, yvalues_shams = [], []
    c0 = (np.array(Z_group[:,4], dtype=float) > 1)
    # c0 = np.logical_and((np.array(Z_group[:,4], dtype=float) < 4), (np.array(Z_group[:,4], dtype=float) > 1))
    
    # 예제 조건 및 데이터
    conditions = [c0, c1, g_vns, inlow]
    vix = np.where(np.logical_and.reduce(conditions))[0]

    yvalues_vns = np.median(np.array(X_chmean)[vix], axis=0)
    sem_vns = sem(np.array(X_chmean)[vix], axis=0, nan_policy='omit')
    yvalues_smoothing_vns = moving_average_smoothing(yvalues_vns, 20)
    
    barplot[cue][ans][0] = np.array(X_chmean)[vix]
    
    plt.plot(xaxis[s:e], yvalues_smoothing_vns[s:e], label=cue + '_VNS_' + str(anslabel), linewidth=2)
    plt.fill_between(xaxis[s:e], 
                      (yvalues_smoothing_vns - sem_vns)[s:e], 
                      (yvalues_smoothing_vns + sem_vns)[s:e], 
                      color='blue', alpha=0.3)
    
    conditions = [c0, c1, g_sham, inlow]
    vix = np.where(np.logical_and.reduce(conditions))[0]
    yvalues_sham = np.median(np.array(X_chmean)[vix], axis=0)
    sem_sham = sem(np.array(X_chmean)[vix], axis=0, nan_policy='omit')
    yvalues_smoothing_sham = moving_average_smoothing(yvalues_sham, 20)
    
    barplot[cue][ans][1] = np.array(X_chmean)[vix]
    
    plt.plot(xaxis[s:e], yvalues_smoothing_sham[s:e], label=cue + '_SHAM_' + str(anslabel), linewidth=2)
    plt.fill_between(xaxis[s:e], 
                     (yvalues_smoothing_sham - sem_sham)[s:e], 
                     (yvalues_smoothing_sham + sem_sham)[s:e], 
                     color='red', alpha=0.3)

#%% ERP 추가분석2

SR = 250
s, e = int(round(SR*(0.2+0.3))), int(round(SR*(0.2+0.4))) # 200~350 ms



stimulus_types = np.array(['neutral', 'congruent', 'incongruent'])
ingroup = np.logical_or((Z_group[:,-1] == 'sham'), (Z_group[:,-1] == 'VNS'))
# incorrect = (Z_group[:,1] == '맞음')

vixsave, vixz = [], []
group_yhat_y = []
mskey_list = list(set(Z_group[:,3]))

X_semean = []
Z_semean = []

# means_stds_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\means_stds.pkl'
# with open(means_stds_path, 'rb') as f:
#     loaded_means_stds = pickle.load(f)
    
# def calculate_zscore(data, mean, std):
#     return (data - mean) / std

# def min_max_scale(data, feature_range=(-0.4, 1.07)):
#     data_min, data_max = feature_range
#     scaled_data = (data - data_min) / (data_max - data_min)

#     return scaled_data


# session = [[0,1], [2,3], [4,5]]

for i in range(len(mskey_list)):
    for j in range(len(stimulus_types)):
        for t in range(6):
            stimulus = stimulus_types[j]
            msid = mskey_list[i]
            
            c0 = (Z_group[:,3] == msid)
            c1 = (Z_group[:,0] == stimulus)
            # c2 = (Z_group[:,4] == str(se))
            c2 = Z_group[:,4] == str(t)
            
            conditions = [c0, c1, c2, ingroup]
            vix = np.where(np.logical_and.reduce(conditions))[0]
            if len(vix) > 0:
                # mrt = np.nanmedian(np.array(Z_group[vix][:,2], dtype=float))
                
                # acc = np.nanmean(Z_group[vix][:,1] == '맞음')
                # print(acc)
                # rt = np.nanmedian(np.array(Z_group[vix][:,2], dtype=float))
                
                # if stimulus == 'neutral': ix1, ix2 = 0, 3
                # if stimulus == 'congruent': ix1, ix2 = 1, 4
                # if stimulus == 'incongruent': ix1, ix2 = 2, 5
                           
                # mean = loaded_means_stds[ix1][0]
                # std = loaded_means_stds[ix1][1]
                # z_score_rt = calculate_zscore(rt, mean, std)
                
                # mean = loaded_means_stds[ix2][0]
                # std = loaded_means_stds[ix2][1]
                # z_score_acc = calculate_zscore(acc, mean, std)
                
                # # tscore = -z_score_rt * 0.5 + z_score_acc * 0.5
                # tscore = acc
                
                
                X_semean.append(np.mean(X_chmean[vix], axis=0))
                Z_semean.append([stimulus, '맞음', msid, t, Z_group[vix][0][-1]])
  
X_semean, Z_semean = np.array(X_semean), np.array(Z_semean)
    
#%%
cues = ['neutral', 'congruent', 'incongruent']
ans = '맞음'
for cue in cues:
    plt.figure(figsize=(6, 4))
    c1 = (Z_semean[:,0] == cue)
    # for ans in ['맞음', '틀림']:
    for t in range(6): 
        g_vns = (Z_semean[:,-1] == 'VNS')
        g_sham = (Z_semean[:,-1] == 'sham')
        
        vns_plots = []
        sham_plots = []
        
        SR = 250
        s,e = int((0)*SR), int((0.20+1.5)*SR)
        
        yvalues_vnss, yvalues_shams = [], []
        c0 = (np.array(Z_semean[:,3], dtype=float) == t)
        # c0 = np.logical_and((np.array(Z_group[:,4], dtype=float) < 4), (np.array(Z_group[:,4], dtype=float) > 1))
        
        # 예제 조건 및 데이터
        conditions = [c0, c1, g_vns]
        vix = np.where(np.logical_and.reduce(conditions))[0]
    
        yvalues_vns = np.mean(np.array(X_chmean)[vix], axis=0)
        sem_vns = sem(np.array(X_chmean)[vix], axis=0, nan_policy='omit')
        yvalues_smoothing_vns = moving_average_smoothing(yvalues_vns, 20)
        
        barplot[cue][0].append(np.array(X_chmean)[vix])
        
        plt.plot(xaxis[s:e], yvalues_smoothing_vns[s:e], \
                  label=cue + '_VNS_' + str(anslabel), linewidth=2, c=colors1[t])
        # plt.fill_between(xaxis[s:e], 
        #                   (yvalues_smoothing_vns - sem_vns)[s:e], 
        #                   (yvalues_smoothing_vns + sem_vns)[s:e], 
        #                   color='blue', alpha=0.3)
        
        conditions = [c0, c1, g_sham]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        yvalues_sham = np.mean(np.array(X_chmean)[vix], axis=0)
        sem_sham = sem(np.array(X_chmean)[vix], axis=0, nan_policy='omit')
        yvalues_smoothing_sham = moving_average_smoothing(yvalues_sham, 20)
        
        barplot[cue][1].append(np.array(X_chmean)[vix])
        
        # plt.plot(xaxis[s:e], yvalues_smoothing_sham[s:e], \
        #           label=cue + '_SHAM_' + str(anslabel), linewidth=2, c=colors2[t])
        # plt.fill_between(xaxis[s:e], 
        #                   (yvalues_smoothing_sham - sem_sham)[s:e], 
        #                   (yvalues_smoothing_sham + sem_sham)[s:e], 
        #                   color='red', alpha=0.3)



#%%
from scipy.stats import pearsonr
SR = 250
s, e = int(round(SR*(0.2+0.25))), int(round(SR*(0.2+0.55))) # 200~350 ms

c1 = (Z_semean[:,6] == 'sham')
# ans = '맞음'

# for ans in ['맞음']:
for cue in ['neutral', 'congruent', 'incongruent']:
    
# for t in [[0,1], [2,3], [4,5]]:
    plt.figure()
    c2 = (Z_semean[:,0] == cue)
    # c3 = np.logical_or(Z_semean[:,4] == str(t[0]), Z_semean[:,4] == str(t[1]))
    c3 = (np.array(Z_semean[:,4], dtype=float) > 0)
    # c3 = Z_semean[:,4] == str(t[0])
    conditions = [c2, c3]
    vix = np.where(np.logical_and.reduce(conditions))[0]
    
    print(len(vix))
    N2b = np.nanmedian(X_semean[vix][:,s:e], axis=1)
    rt = np.array(Z_semean[vix][:,2], dtype=float)
    
    cc1 = np.logical_and(np.isnan(rt)==0, np.isnan(N2b)==0)
    cc2 = np.logical_and(N2b < 100000, N2b > -100000)
    andvix = np.logical_and(cc1, cc2)
    
    corr1, corr2 = np.array(N2b[andvix]), np.array(rt[andvix])
    
    corr, p_value = pearsonr(corr1, corr2)
    # print(cue, t)
    print(cue + str(t), corr)
    plt.scatter(corr1, corr2, s=2, label = str(t))
    plt.title(cue + str(t))
    plt.legend()

"""
1. sham group한정하여, [4,5] session 한정할 때, 위 corr 보임 -> 굉장히 강함
그러니깐, [4,5]가 가장 negativity가 높으면서도, 그 와중에 negativity 낮은 사람이 잘한다는 역설적인 결과


2. rt가 아닌 종합 스코어로도 볼필요 있음. >>> 숙제
3. VNS group에서는 regression line이 역수가 됨. -> ??
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

SR = 250
s, e = int(round(SR*(0.2+0.25))), int(round(SR*(0.35+0.25))) # 200~350 ms

c1 = (Z_semean[:,6] == 'sham')
ans = '맞음'

# for ans in ['맞음']:
# for cue in ['neutral', 'congruent', 'incongruent']:
for t in [[0,1], [2,3], [4,5]]:
    c0 = (Z_semean[:,1] == ans)
    c2 = (Z_semean[:,0] == cue)
    c3 = np.logical_or(Z_semean[:,4] == str(t[0]), Z_semean[:,4] == str(t[1]))
    conditions = [c0, c1, c3]
    vix = np.where(np.logical_and.reduce(conditions))[0]
    
    print(len(vix))
    N2b = np.median(X_semean[vix][:,s:e], axis=1)
    
    rt = np.array(Z_semean[vix][:,2], dtype=float)
    
    corr1, corr2 = np.array(N2b), np.array(rt)
    
    print('N2b', np.mean(corr1))
    
    corr, p_value = pearsonr(corr1, corr2)
    print(cue, t)
    print(corr)
    
    # Calculate means
    mean_corr1 = np.mean(corr1)
    mean_corr2 = np.mean(corr2)
    
    # Scatter plot with regression line
    plt.figure()
    plt.scatter(corr1, corr2, label=str(t), alpha=0.5, edgecolors='none')
    
    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(corr1, corr2)
    xdummy = np.arange(-1, 1.1, 0.1)
    plt.plot(xdummy, slope*xdummy + intercept, label='regression line', color='#FFA07A', linewidth=2, alpha=0.5)
    
    plt.xlim(-1, 1)
    plt.ylim(400, 1200)
    
    # Plot mean lines
    plt.axhline(mean_corr2, color='gray', linestyle='--', linewidth=1)
    plt.axvline(mean_corr1, color='gray', linestyle='--', linewidth=1)

    # Add text annotations for means
    plt.text(max(corr1), mean_corr2, f'Mean RT: {mean_corr2:.2f}', color='gray', horizontalalignment='right', verticalalignment='bottom')
    plt.text(mean_corr1, max(corr2), f'Mean N2b: {mean_corr1:.2f}', color='gray', horizontalalignment='right', verticalalignment='bottom')
    plt.text(-0.9, 1150, f'R² = {r_value**2:.2f}', color='black', horizontalalignment='left')

    # plt.text(mean_corr1, mean_corr2, f'({mean_corr1:.2f}, {mean_corr2:.2f})', color='gray', horizontalalignment='right', verticalalignment='bottom')

    plt.xlabel('N2b')
    plt.ylabel('RT')
    plt.title(f'Scatter plot and regression line for {t}')
    plt.legend()
    
    if True: # figure save codes
        fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
        fsave2 = fsave + '\\' + cue + '_ERP_corr_sham' + str(t) + '.png'
        plt.tight_layout()
        plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
        plt.show()
    




#%%

tscore_save
msdict_array[0]

g0 = msdict_array[:,15]=='VNS'
g1 = msdict_array[:,15]=='sham'


SR = 250
s, e = int(round(SR*(0.2+0.25))), int(round(SR*(0.35+0.25))) # 200~350 ms

c1 = (Z_semean[:,6] == 'VNS')
ans = '맞음'

# for ans in ['맞음']:
for cue in ['neutral', 'congruent', 'incongruent']:
    for t in [[4,5]]:
        c0 = (Z_semean[:,1] == ans)
        c2 = (Z_semean[:,0] == cue)
        c3 = np.logical_or(Z_semean[:,4] == str(t[0]), Z_semean[:,4] == str(t[1]))
        conditions = [c0, c3]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        
        print(len(vix))
        N2b = np.median(X_semean[vix][:,s:e], axis=1)
        # print(ans, cue, np.mean(N2b))
        
        
        c1 = (msdict_array[:,15] == 'VNS')
        c3 = np.logical_or(msdict_array[:,13] == t[0], msdict_array[:,13] == t[1])
        conditions = [c1, c3]
        vix = np.where(np.logical_and.reduce(conditions))[0]
        tscore = tscore_save[vix]
        
        corr1, corr2 = np.array(N2b), np.array(tscore)
        corr, p_value = pearsonr(corr1, corr2)
        print(cue, t)
        print(corr)
        plt.scatter(corr1, corr2, s=2)












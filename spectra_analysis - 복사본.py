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

# SR = 250  # 샘플링 레이트
# T = 1.0 / SR  # 샘플 간격
# L = 2500  # 데이터 길이 (10초 분량)
# t = np.linspace(0.0, L * T, L, endpoint=False)  # 시간 벡터
# # 10Hz 사인파 더미 데이터 생성
# frequency = 10  # 10Hz
# amplitude = 1.0  # 진폭
# data = amplitude * np.sin(2 * np.pi * frequency * t)


def fft_calc(s=None, e=None, eeg_df=eeg_df, betas=[1, 1]):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq
    
    SR = 250
    tix = eeg_df[:,-1]
    
    vix = np.where(np.logical_and((tix > s), (tix < e)))[0]
    vt = tix[vix]
    vix2 = vix[np.argsort(vt)]
    eeg_df2 = np.array(eeg_df[vix2])
    
    # X = []
    # tlen = int(SR*220)
    # msbins = np.arange(0, eeg_df2.shape[0]-tlen, int(SR * 60), dtype=int)
    # for j in range(len(msbins)):
    #     xtmp2 = eeg_df2[msbins[j]:msbins[j]+tlen,:][:,[0,1]]
    #     if xtmp2.shape[0] == tlen:
    #         xtmp3 = []
    #         for ch in range(2):
    #             xtmp3.append(xtmp2[:,ch]/np.mean(xtmp2[:, ch]))
            # X.append(xtmp3)
             
    X = np.array(eeg_df2[:,[0,1]])

    DATA_LENGTH = len(X)
    beta1, beta2 = betas
    
    mssave = []
    # for i2 in range(len(X)):
    eeg_data = (X[:,0] * beta1) + (X[:,1] * beta2)
    yf = fft(eeg_data)
    xf = fftfreq(DATA_LENGTH, 1/SR)
    
    # 0 ~ 40 Hz 대역의 파워 스펙트럼 추출
    xf_positive = xf[:DATA_LENGTH // 2]
    yf_positive = np.abs(yf[:DATA_LENGTH // 2]) ** 2
    
    # 주파수 해상도 확인
    # frequency_resolution = SR / DATA_LENGTH 
    # print(f"Frequency Resolution: {frequency_resolution} Hz")
    
    # 2 ~ 40 Hz 대역의 주파수와 파워 스펙트럼 추출
    s_fi, e_if = 8, 12
    
    mask = (xf_positive >= s_fi) & (xf_positive <= e_if)
    xf_filtered = xf_positive[mask]
    yf_filtered = yf_positive[mask]
    
    # 0.1 Hz 단위로 주파수 대역을 분류 (2 ~ 40 Hz)
    target_freqs_01 = np.arange(s_fi, e_if+0.01, 0.1)
    power_spectrum_01 = np.zeros_like(target_freqs_01)
    
    for i, freq in enumerate(target_freqs_01):
        idx = np.argmin(np.abs(xf_filtered - freq))
        power_spectrum_01[i] = yf_filtered[idx]
    
    # 0.5 Hz 단위로 재분류
    target_freqs_05 = np.arange(s_fi, e_if+0.01, 0.5)
    power_spectrum_05 = np.zeros_like(target_freqs_05)
    
    for i in range(len(target_freqs_05)):
        start_idx = i * 5
        end_idx = start_idx + 5
        power_spectrum_05[i] = np.mean(power_spectrum_01[start_idx:end_idx])

    xin = power_spectrum_05 /np.sum(power_spectrum_05)
        # mssave.append(xin)

        # plt.plot(target_freqs_05, xin)

    return np.array([np.array(xin)])


def apply_morlet_wavelet(data, sr):
    import numpy as np
    from scipy.signal import morlet2, cwt
    
    frequencies = np.arange(4, 51)  # 4~50 Hz
    # data - (ch, t)
    power = []
    phase = []
    for channel_data in data:
        widths = sr / (2 * frequencies * np.pi)
        cwt_result = cwt(channel_data, morlet2, widths)
        power.append(np.abs(cwt_result)**2)
        phase.append(np.angle(cwt_result))
    return np.array(power), np.array(phase)

#%%

beta1, beta2 = 1, 1

beta1_range = np.arange(-5, 5.1, 0.5)
beta2_range = np.arange(-5, 5.1, 0.5)

save_matrix = np.zeros((len(beta1_range), len(beta1_range)))
for b1, beta1 in enumerate(beta1_range):
    for b2, beta2 in enumerate(beta2_range):
        
        #%%
        
        psd_pre_save, psd_post_save =  [], []
        idlist = list(experiment_data.keys())
        for id_i in range(len(idlist)):
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
            psd_pre = fft_calc(s=s, e=e, eeg_df=eeg_df, betas=[beta1, beta2])
            
            # print(psd_pre.shape)
            # plt.plot(np.mean(psd_pre , axis=0))
        
            # post-baseline
            s = last_cue + (0.5*60)
            e = last_cue + (5.5*60)
            psd_post = fft_calc(s=s, e=e, eeg_df=eeg_df, betas=[beta1, beta2])
        
            if len(psd_pre_save) == 0:
                psd_pre_save = np.array(psd_pre)
                psd_post_save = np.array(psd_post)
            
            else:
                psd_pre_save = np.concatenate((psd_pre_save, np.array(psd_pre)), axis=0)
                psd_post_save = np.concatenate((psd_post_save, np.array(psd_post)), axis=0)
            
   
        median_psd = np.mean(psd_pre_save, axis=0)
        median_psd_2 = np.mean(psd_post_save, axis=0)
        
        loss_maximize = np.mean(np.abs(median_psd - median_psd_2))
        save_matrix[b1,b2] = loss_maximize
        print(beta1, beta2, loss_maximize)
# print(psd_pre_save.shape, psd_post_save.shape)
#%%
s_fi, e_if = 4, 8
target_freqs_05 = np.linspace(s_fi, e_if, psd_pre_save.shape[1])

median_psd = np.mean(psd_pre_save, axis=0)
sem_psd = sem(psd_pre_save, axis=0)
plt.plot(target_freqs_05, median_psd, label='First Interval')
plt.fill_between(target_freqs_05, median_psd - sem_psd, median_psd + sem_psd, alpha=0.3)

median_psd_2 = np.mean(psd_post_save, axis=0)
sem_psd_2 = sem(psd_post_save, axis=0)

plt.plot(target_freqs_05, median_psd_2, label='Second Interval')
plt.fill_between(target_freqs_05, median_psd_2 - sem_psd_2, median_psd_2 + sem_psd_2, alpha=0.3)

# 플롯 설정
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (PSD)')
plt.legend()
plt.title('PSD with Median and SEM')
plt.show()



#%% EEG data load

# import glob
# from tqdm import tqdm 

# directory_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\eegdata'
# pkl_files = glob.glob(os.path.join(directory_path, '*.pkl'))

# for j in tqdm(range(len(pkl_files))):
#     with open(pkl_files[j], 'rb') as file:
#         msdict = pickle.load(file)
        
#     if j == 0:
#         eeg_df = np.array(msdict)
#     if j > 0:
#         eeg_df = np.concatenate((eeg_df, np.array(msdict)), axis=0)

# print(eeg_df.shape)



#%%




#%%



# s = first_cue - (7*60)
# e = first_cue - (2*60)
# psd = fft_calc(s=s, e=e, eeg_df=eeg_df)
# plt.plot(np.median(psd, axis=0))


























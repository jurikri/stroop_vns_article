# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:38:56 2024

@author: PC
"""

import os
import sys
sys.path.append(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis')
sys.path.append(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl')
import stroop_pkl_to_exceldb
import ranking_board_backend3
import pandas as pd
import pickle
import numpy as np
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode')
import msFunction
import stroop_score_calgen
import matplotlib.pyplot as plt
from scipy import stats

# import pandas as pd
#%%

# 지정된 경로
strooppath = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\rawdata\stroopdata'

# 결과를 저장할 리스트
folders_list = []
pkl_files_list = []

# 모든 폴더를 리스트업
for root, dirs, files in os.walk(strooppath):
    for dir_name in dirs:
        folders_list.append(os.path.join(root, dir_name))

# 각 폴더의 block1과 block2 폴더 내의 *.pkl 파일을 찾음
for folder in folders_list:
    for block in ['block1', 'block2']:
        block_path = os.path.join(folder, block)
        if os.path.exists(block_path):
            for root, dirs, files in os.walk(block_path):
                for file in files:
                    if file.endswith('.pkl'):
                        pkl_files_list.append(os.path.join(root, file))

pkl_files_list = sorted(pkl_files_list)

preexist = []
iddict = {}
for i in range(len(pkl_files_list)):
    # 상위 경로 식별
    parent_dir = os.path.dirname(pkl_files_list[i])  # block2 폴더 경로
    grandparent_dir = os.path.dirname(parent_dir)  # 202405261555_LJH 폴더 경로
    
    # block2와 202405261555_LJH 폴더 이름 추출
    block_folder = os.path.basename(parent_dir)
    experiment_folder = os.path.basename(grandparent_dir)
    
    print("Block Folder:", block_folder)
    print("Experiment Folder:", experiment_folder)
    
    date_part, identifier = experiment_folder.split('_')
    # date_part2 = date_part[:8]
    print("Date Part:", date_part)
    print("Identifier:", identifier)
    msid = date_part + '_' + identifier\
    
    if not msid in list(iddict.keys()): 
        iddict[msid] = {}
        iddict[msid]['block1'] = []
        iddict[msid]['block2'] = []
    
    iddict[msid][block_folder].append(pkl_files_list[i])

"""
위쪽 경로식별 코드가 깔끔하진 않으나 작동하므로 그냥 둔다.
"""

#%%

psave_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl'

keylist = list(iddict.keys())
for n in range(len(keylist)):
    # t = 'block1'
    for t in ['block1', 'block2']:
        ablock_stroop_paths = list(iddict[keylist[n]][t])
        for k in range(len(ablock_stroop_paths)):
            ranking_board_backend3.msmain\
                (block_path=os.path.dirname(ablock_stroop_paths[k]), \
                 psave_path=psave_path)
                    
stroop_pkl_to_exceldb.msmain()
    
"""
C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl
에 pkl로 개별 저장 후,
stroop_score_db.xlsx 에 시각화 가능하게 저장하는 방식
"""


df  = pd.read_excel(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db.xlsx')
pickle_file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db.pkl'
df.to_pickle(pickle_file_path)
print(f"DataFrame successfully saved to {pickle_file_path}")


#%%
pload = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db.pkl'
with open(pload, 'rb') as file:
    df = pickle.load(file)

# 'msid2' 컬럼을 '_'를 기준으로 분리하여 'date'와 'name' 컬럼을 생성
df['name'] = df['msid2'].str.split('_', n=1, expand=True)[1]
df['date'] = df['msid2'].str[:8]
dataset_counts = df.groupby(['date', 'name']).size()
problematic_datasets = dataset_counts[dataset_counts != 6]
df_filtered = df[~df.set_index(['date', 'name']).index.isin(problematic_datasets.index)]
filtered_dataset_counts = df_filtered.groupby(['date', 'name']).size()
assert (filtered_dataset_counts == 6).all(), "Some datasets still do not meet the requirement."

# DataFrame을 'date', 'name', 'msid2' 순으로 정렬
df_filtered['session'] = df_filtered.groupby(['date', 'name']).cumcount()
# 새로운 DataFrame df2 생성
df2 = df_filtered.copy()
df2['ID'] = df2['date'] + '_' + df2['name']
msdict = pd.DataFrame(df2)


#%% 나이 계산
from datetime import datetime
eload = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\임상시험 그룹 현황 240624.xlsx'
df_age = pd.read_excel(eload, dtype={'만 나이': str})

# 현재 날짜
current_date = datetime.now()

# 만나이를 개월수 단위로 계산하는 함수
def calculate_age_in_months(birth_date_str):
    if int(birth_date_str[:2]) <= 24:
        year_prefix = '20'
    else:
        year_prefix = '19'
    birth_date = datetime.strptime(year_prefix + birth_date_str, '%Y%m%d')
    age_in_months = (current_date.year - birth_date.year) * 12 + current_date.month - birth_date.month
    return age_in_months
# 각 생년월일에 대해 개월수 단위 나이를 계산
ages_in_months = [calculate_age_in_months(date) for date in np.array(df_age['만 나이'])]

#%%

df = pd.read_excel(eload)
msids = np.array(df['ID'])
group_info = np.array(df['그룹'])

label_info = []
for i in range(len(msids)):
    if type(msids[i]) == str:
        label = None
        if group_info[i] ==  'A) VNS 실험군':
            label = 'VNS'    
        elif group_info[i] ==  'B) sham 대조군':
            label = 'sham'
        label_info.append([msids[i], label])
label_info = np.array(label_info)
np.save(r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\label_info.npy', label_info, allow_pickle=True)

# Convert the label_info array to a dictionary for easier lookup
label_dict = dict(label_info)
# Add a new column 'group' to the DataFrame based on the ID matching
msdict['group'] = msdict['ID'].map(label_dict)

output_file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db2.pkl'
with open(output_file_path, 'wb') as file:
    pickle.dump(msdict, file)


#%%


pload = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db2.pkl'
with open(pload, 'rb') as file:
    msdict = pickle.load(file)

msdict_array = np.array(msdict)

# msdict['tscore_save'] = tscore_save
# file_path = 'example.xlsx'
# msdict.to_excel(file_path, index=False)


#%%
import stroop_score_calgen
import msFunction


# plt.scatter(np.mean(data, axis=1), tscore_save)

i = 0
tscore_save = []
for i in range(len(msdict_array)):
    data = np.array(msdict_array[i][[3,6,9,2,5,8]], dtype=float)  
    score = stroop_score_calgen.stroop_tscore(data, weights = [[0.4, 0.6], [0.25, 0.25, 0.5]])
    tscore_save.append(score)
tscore_save = np.array(tscore_save)

# plt.scatter(tscore_save, np.exp(tscore_save ))
# tscore_save = np.exp(tscore_save * 2)

g0 = msdict_array[:,15]=='VNS'
g1 = msdict_array[:,15]=='sham'

print(np.mean(tscore_save[g0]), np.mean(tscore_save[g1]))

t_stat, p_value = stats.ttest_ind(tscore_save[g0], tscore_save[g1])
print(i)
print("t-statistic:", t_stat)
print("p-value:", p_value)

trial, group = 6, 2
mssave = msFunction.msarray([trial, group])
mssave_line_vns, mssave_line_sham = [], []  
for trial in range(6):
    tix = np.array(msdict_array[:,13], dtype=int)==trial
    
    vix = np.logical_and(g0, tix)
    rt1 = tscore_save[vix]
    mssave[trial][0] = rt1
    
    vix = np.logical_and(g1, tix)
    rt2 = tscore_save[vix]
    mssave[trial][1] = rt2
    
    mssave_line_vns.append(rt1)
    mssave_line_sham.append(rt2)
    
#%%  plot - 2 - trial 두개 합치기

# VNS와 sham 그룹의 평균값과 SEM을 계산 (x축을 3으로 축소)
vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

# trials = mssave[0]
i = [0,1]
for i in [[0,1], [2,3], [4,5]]:
    # trials = np.mean(np.array([mssave[i[0]], mssave[i[1]]]), axis=0) # 평균내기
    
    # trials = [np.concatenate((mssave[i[0]][0], mssave[i[1]][0]), axis=0),\
    #           np.concatenate((mssave[i[0]][1], mssave[i[1]][1]), axis=0)] # 같이 취급하기
        
    trials = [np.mean(np.array([mssave[i[0]][0], mssave[i[1]][0]]), axis=0),\
              np.mean(np.array([mssave[i[0]][1], mssave[i[1]][1]]), axis=0)] # 같이 취급하기
    
    print(len(trials[0]), len(trials[1]))
    vns_means.append(np.median(trials[0]))
    sham_means.append(np.median(trials[1]))
    
    vns_sems.append(np.std(trials[0]) / np.sqrt(len(trials[0])))
    sham_sems.append(np.std(trials[1]) / np.sqrt(len(trials[1])))

x = np.arange(1, 4)
width = 0.35

fig, ax = plt.subplots()
# bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
# bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)
bars1 = ax.bar(x - width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)
bars2 = ax.bar(x + width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)

# 그래프 설정
ax.set_xlabel('Session')
ax.set_ylabel('Stroop Task Intergrated Score')
ax.set_title('Stroop Task Performance by Session')
ax.set_xticks(x)
ax.set_xticklabels(x)
# ax.set_ylim(4, 7)  # Y축 범위 설정
ax.legend()

if False:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure1_Strooptask_performance.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

# 그래프 보여주기
plt.show()

for i in [[0,1], [2,3], [4,5]]:
    # trials = np.mean(np.array([mssave[i[0]], mssave[i[1]]]), axis=0) # 평균내기
    
    trials = [np.concatenate((mssave[i[0]][0], mssave[i[1]][0]), axis=0),\
              np.concatenate((mssave[i[0]][1], mssave[i[1]][1]), axis=0)] # 같이 취급하기

    t_stat, p_value = stats.ttest_ind( trials[0], trials[1])
    print(i)
    print("t-statistic:", t_stat)
    print("p-value:", p_value)


#%%
msdict = {}
for session in range(len(mssave)):
    seid = 'trial_' + str(session)
    msdict[seid] = {}
    msdict[seid]['VNS'] =  mssave[session][0]
    msdict[seid]['sham'] =  mssave[session][1]
    
psave = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result_data\stroop_performance1.pkl'
with open(psave, 'wb') as f:
    pickle.dump(msdict, f)
    
    
# Updating the dataframe to include mean, median, SEM, and N
data_for_excel = {
    'Session': [],
    'Group': [],
    'Mean': [],
    'Median': [],
    'SEM': [],
    'N': []
}

for session, groups in msdict.items():
    vns_data = groups['VNS']
    sham_data = groups['sham']
    
    vns_mean = np.mean(vns_data)
    vns_median = np.median(vns_data)
    vns_sem = np.std(vns_data) / np.sqrt(len(vns_data))
    vns_n = len(vns_data)
    
    sham_mean = np.mean(sham_data)
    sham_median = np.median(sham_data)
    sham_sem = np.std(sham_data) / np.sqrt(len(sham_data))
    sham_n = len(sham_data)
    
    data_for_excel['Session'].extend([session, session])
    data_for_excel['Group'].extend(['VNS', 'sham'])
    data_for_excel['Mean'].extend([vns_mean, sham_mean])
    data_for_excel['Median'].extend([vns_median, sham_median])
    data_for_excel['SEM'].extend([vns_sem, sham_sem])
    data_for_excel['N'].extend([vns_n, sham_n])

df = pd.DataFrame(data_for_excel)
# Saving to Excel file
file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result_data\stroop_performance1.xlsx'
df.to_excel(file_path, index=False)

    
#%% plot - 1
import matplotlib.pyplot as plt

vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

vns_nmr = np.array(mssave[0][0])
sham_nmr = np.array(mssave[0][1])

# trials = mssave[0]
for trials in mssave:
    vns = trials[0] # - vns_nmr
    shame = trials[1] # - sham_nmr
    
    vns_means.append(np.mean(vns))
    sham_means.append(np.mean(shame))
    
    vns_sems.append(np.std(vns) / np.sqrt(len(vns)))
    sham_sems.append(np.std(shame) / np.sqrt(len(shame)))
    
x = np.arange(1, 7)

# 바의 너비
width = 0.35
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)

# 그래프 설정
ax.set_xlabel('Trial')
ax.set_ylabel('Value')
ax.set_title('VNS vs Sham Data by Trial')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

# 그래프 보여주기
plt.show()

#%%
from scipy.stats import pearsonr

idset = list(set(msdict_array[:,14]))
corr = []
for m in range(len(idset)):
    vix = np.where(np.array(msdict_array[:,14]) == idset[m])[0]
    idrow = np.where(label_info[:,0] == idset[m])[0]
    if len(idrow) > 0:
        # if ages_in_months[idrow[0]]/12 < 32:
        corr.append([ages_in_months[idrow[0]]/12, np.mean(tscore_save[vix])])
    else:
        print(m, idset[m])

corr = np.array(corr)

pearson_corr, _ = pearsonr(corr[:, 0], corr[:, 1])
print(f"Pearson correlation coefficient: {pearson_corr}")

# 산점도 생성
plt.scatter(corr[:, 0], corr[:, 1])
plt.title(f'Scatter Plot (Pearson correlation: {pearson_corr:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

z = np.polyfit(corr[:, 0], corr[:, 1], 1)
p = np.poly1d(z)
plt.plot(corr[:, 0], p(corr[:, 0]), "r--", label='Trendline')

plt.legend()
plt.grid(True)
plt.show()


#%%

g0 = msdict_array[:,15]=='VNS'
g1 = msdict_array[:,15]=='sham'

# acc, rt, 2 ,3 이고 +3 씩

trial, group = 6, 2
mssave_rt = msFunction.msarray([trial, group, 6])

for ix, i in enumerate([3,6,9, 2,5,8]):
    for trial in range(6):
        tix = np.array(msdict_array[:,13], dtype=int)==trial
        
        if i in [3,6,9]:
            vix = np.logical_and(g0, tix)
            rt1 = np.array(msdict_array[vix, i], dtype=float)
            mssave_rt[trial][0][ix] = rt1
            
            vix = np.logical_and(g1, tix)
            rt2 = np.array(msdict_array[vix, i], dtype=float)
            mssave_rt[trial][1][ix] = rt2
            
        elif i in [2,5,8]:
            vix = np.logical_and(g0, tix)
            rt1 = np.array(msdict_array[vix, i], dtype=float)
            mssave_rt[trial][0][ix] = 1 - rt1
            
            vix = np.logical_and(g1, tix)
            rt2 = np.array(msdict_array[vix, i], dtype=float)
            mssave_rt[trial][1][ix] = 1 - rt2
               
#% [0,1], [2,3], [4,5] session 합쳐서 3 se로

import matplotlib.pyplot as plt

msdict_save = {}
dict_name = ['neutral_rt', 'congruent_rt', 'incongruent_rt', \
             'neutral_acc', 'congruent_acc', 'incongruent_acc']

mssave_rt_2se = msFunction.msarray([3, group, 6])       
for fn in range(6):
    msdict_save[dict_name[fn]] = {}
    for i in [[0,1], [2,3], [4,5]]:
        trials = [np.mean(np.array([mssave_rt[i[0]][0][fn], mssave_rt[i[1]][0][fn]]), axis=0),\
                  np.mean(np.array([mssave_rt[i[0]][1][fn], mssave_rt[i[1]][1][fn]]), axis=0)] # 같이 취급하기

        msdict_save[dict_name[fn]][str(i)] = {}
        print(len(trials[0]))
        msdict_save[dict_name[fn]][str(i)]['VNS'] = trials[0]
        msdict_save[dict_name[fn]][str(i)]['sham'] = trials[1]

psave = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result_data\stroop_rt_acc.pkl'
with open(psave, 'wb') as f:
    pickle.dump(msdict_save, f)
    
import numpy as np
import pandas as pd

# Helper function to compute statistics
def compute_stats(values):
    mean = np.mean(values)
    median = np.median(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))  # Standard error of the mean
    N = len(values)
    return mean, median, sem, N

# Create a dictionary to hold the data for the dataframe
stats_dict = {
    'Combination': [],
    'Session': [],
    'Group': [],
    'Mean': [],
    'Median': [],
    'SEM': [],
    'N': []
}

# Extracting and calculating statistics for each combination
for combo in ['neutral_rt', 'neutral_acc', 'congruent_rt', 'congruent_acc', 'incongruent_rt', 'incongruent_acc']:
    for session, key in zip(['Session 1', 'Session 2', 'Session 3'], ['[0, 1]', '[2, 3]', '[4, 5]']):
        for group in ['VNS', 'sham']:
            values = msdict_save[combo][key][group]
            mean, median, sem, N = compute_stats(values)
            stats_dict['Combination'].append(combo)
            stats_dict['Session'].append(session)
            stats_dict['Group'].append(group)
            stats_dict['Mean'].append(mean)
            stats_dict['Median'].append(median)
            stats_dict['SEM'].append(sem)
            stats_dict['N'].append(N)

# Convert the dictionary to a DataFrame
stats_df = pd.DataFrame(stats_dict)

# Save the DataFrame to an Excel file
file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result_data\stroop_rt_acc.xlsx'
stats_df.to_excel(file_path, index=False)

#%%
import matplotlib.pyplot as plt

mssave = []
for se in [[0, 1], [2, 3], [4, 5]]:
    vns_diff = msdict_save['incongruent_rt'][str(se)]['VNS'] - msdict_save['congruent_rt'][str(se)]['VNS']
    sham_diff = msdict_save['incongruent_rt'][str(se)]['sham'] - msdict_save['congruent_rt'][str(se)]['sham']
    mssave.append([sham_diff, vns_diff])

vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

sham_nmr = np.array(mssave[0][0])
vns_nmr = np.array(mssave[0][1])

# trials = mssave[0]
for trials in mssave:
    vns = trials[1] # - vns_nmr
    shame = trials[0] # - sham_nmr
    
    vns_means.append(np.mean(vns))
    sham_means.append(np.mean(shame))
    
    vns_sems.append(np.std(vns) / np.sqrt(len(vns)))
    sham_sems.append(np.std(shame) / np.sqrt(len(shame)))
    
x = np.arange(1, 4)

# 바의 너비
width = 0.35
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)

# 그래프 설정
ax.set_xlabel('Trial')
ax.set_ylabel('Value')
ax.set_title('VNS vs Sham Data by Trial')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

# 그래프 보여주기
plt.show()


#%% whisker plot

from scipy import stats
from matplotlib.patches import Patch

dict_name = ['neutral_rt', 'congruent_rt', 'incongruent_rt', \
             'neutral_acc', 'congruent_acc', 'incongruent_acc']
             
figure_title = ['Neutral Cue', 'Congruent Cue', 'Incongruent Cue', \
                'Neutral Cue', 'Congruent Cue', 'Incongruent Cue']
    
sessions = [[0, 1], [2, 3], [4, 5]]
# vns_data, sham_data = [], []

for ix, n in enumerate(dict_name):
    
    # vns_means, vns_sems, sham_means, sham_sems = [] ,[] ,[] ,[]
    vns_data, sham_data = [], []
    for se in sessions:
        # vns_means.append(np.median(msdict_save[n][str(se)]['VNS']))    
        # vns_sems.append(stats.sem(msdict_save[n][str(se)]['VNS']))
        
        vns_data.append(msdict_save[n][str(se)]['VNS'])
        sham_data.append(msdict_save[n][str(se)]['sham'])
        
        # sham_means.append(np.median(msdict_save[n][str(se)]['sham']))    
        # sham_sems.append(stats.sem(msdict_save[n][str(se)]['sham']))
                  
    data = [sham_data[0], vns_data[0], sham_data[1], vns_data[1], sham_data[2], vns_data[2]]
    
    # spacing = 1  # 이 값을 조정하여 간격을 조정할 수 있습니다
    # 파라미터 설정
    whisker_spacing = 0.5
    session_spacing = 2
    figure_width = 10

    # positions 리스트 생성
    positions = []
    current_position = 1
    for i in range(len(sessions)):
        positions.append(current_position)
        positions.append(current_position + whisker_spacing)
        current_position += session_spacing
        
    # 파라미터 설정
    whisker_spacing = 0.3
    session_spacing = 0.7
    figure_width = 4
    
    # positions 리스트 생성
    positions = []
    current_position = 1
    for i in range(len(sessions)):
        positions.append(current_position)
        positions.append(current_position + whisker_spacing)
        current_position += session_spacing
    
    # Figure 생성
    fig, ax = plt.subplots(figsize=(figure_width, 6))
    
    # Box-Whisker plot 생성
    boxprops = dict(color='black', linewidth=1.)
    whiskerprops = dict(color='black', linewidth=1.)
    capprops = dict(color='black', linewidth=1.)
    medianprops = dict(color='black', linewidth=1.)
    
    data = [sham_data[0], vns_data[0], sham_data[1], vns_data[1], sham_data[2], vns_data[2]]
    boxplot = ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True,
                         boxprops=boxprops, whiskerprops=whiskerprops,
                         capprops=capprops, medianprops=medianprops, whis=[0, 100])
    
    # 그룹 구분선을 색상으로 표시
    colors = ['C0', 'C1']
    for patch, color in zip(boxplot['boxes'], [colors[i % 2] for i in range(len(boxplot['boxes']))]):
        patch.set_facecolor(color)
        
    # 레이블 및 제목 설정
    if ix in [0,1,2]: ax.set_ylabel('Reaction time (ms)')
    elif ix in [3,4,5]: ax.set_ylabel('Acuuracy')
    ax.set_title(figure_title[ix])
    
    # X축 틱 및 레이블 설정
    xtick_positions = [positions[2*i] + whisker_spacing / 2 for i in range(len(sessions))]
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels([f'Trial {i+1}' for i in range(len(sessions))])
    
    # 커스텀 레전드 추가
    legend_elements = [Patch(facecolor='C0', edgecolor='black', label='Sham'),
                       Patch(facecolor='C1', edgecolor='black', label='VNS')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if False:
        fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
        fsave2 = fsave + '\\Figure1_basic_' + str(ix) + '.png'
        plt.tight_layout()
        plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
        plt.show()
    
    plt.show()

#%% 기본정보 bar plot

from scipy import stats
from matplotlib.patches import Patch

dict_name = ['neutral_rt', 'congruent_rt', 'incongruent_rt', \
             'neutral_acc', 'congruent_acc', 'incongruent_acc']
             
figure_title = ['Neutral Cue', 'Congruent Cue', 'Incongruent Cue', \
                'Neutral Cue', 'Congruent Cue', 'Incongruent Cue']
    
sessions = [[0, 1], [2, 3], [4, 5]]
# vns_data, sham_data = [], []

for ix, n in enumerate(dict_name):
    
    vns_means, vns_sems, sham_means, sham_sems = [] ,[] ,[] ,[]
    # vns_data, sham_data = [], []
    for se in sessions:
        vns_means.append(np.mean(msdict_save[n][str(se)]['VNS']))    
        vns_sems.append(stats.sem(msdict_save[n][str(se)]['VNS']))
  
        sham_means.append(np.mean(msdict_save[n][str(se)]['sham']))    
        sham_sems.append(stats.sem(msdict_save[n][str(se)]['sham']))
                  
    # data = [sham_data[0], vns_data[0], sham_data[1], vns_data[1], sham_data[2], vns_data[2]]
    
    x = np.arange(1, 4)

    # 바의 너비
    width = 0.35
    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
    bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)

    # 그래프 설정
    ax.set_xlabel('Trial')
    ax.set_ylabel('Value')
    ax.set_title(n)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend()

    # 그래프 보여주기
    if False:
        fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
        fsave2 = fsave + '\\Figure1_basic_' + str(ix) + '.png'
        plt.tight_layout()
        plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
        plt.show()
    
    plt.show()

#%% plot
dict_name = ['neutral_rt', 'congruent_rt', 'incongruent_rt']


# 'neutral_acc', 'congruent_acc', 'incongruent_acc']

dict_name  = ['incongruent_acc']
sessions = [[0,1], [2,3], [4,5]]

vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

for n in dict_name:
    for se in sessions:
        
        trials = []
        trials.append(msdict_save[n][str(se)]['VNS'])
        trials.append(msdict_save[n][str(se)]['sham'])

        vns_means.append(np.mean(trials[0]))
        sham_means.append(np.mean(trials[1]))
        
        vns_sems.append(np.std(trials[0]) / np.sqrt(len(trials[0])))
        sham_sems.append(np.std(trials[1]) / np.sqrt(len(trials[1])))

# 바의 너비
x = np.arange(1, 4)
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)

# 그래프 설정
ax.set_xlabel('Trial')
ax.set_ylabel('Value')
ax.set_title('VNS vs Sham Data by Trial')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

# 그래프 보여주기
plt.show()


#%%
# congruent - incogruent 차이
i = 6
trial, group = 6, 2
mssave = msFunction.msarray([trial, group])
for trial in range(6):
    tix = np.array(msdict_array[:,13], dtype=int)==trial
    
    # 6, 9
    
    vix = np.logical_and(g0, tix)
    rt1 = np.array(msdict_array[vix, 5], dtype=float) # - np.array(msdict_array[vix, 6], dtype=float) 
    mssave[trial][0] = rt1
    
    vix = np.logical_and(g1, tix)
    rt2 = np.array(msdict_array[vix, 5], dtype=float) # - np.array(msdict_array[vix, 6], dtype=float)
    mssave[trial][1] = rt2


#%% plot - 1
import matplotlib.pyplot as plt

vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

vns_nmr = np.array(mssave[0][0])
sham_nmr = np.array(mssave[0][1])

# trials = mssave[0]
for trials in mssave:
    vns = trials[0] # - vns_nmr
    shame = trials[1] # - sham_nmr
    
    vns_means.append(np.mean(vns))
    sham_means.append(np.mean(shame))
    
    vns_sems.append(np.std(vns) / np.sqrt(len(vns)))
    sham_sems.append(np.std(shame) / np.sqrt(len(shame)))
    
    
x = np.arange(1, 7)

# 바의 너비
width = 0.35
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)

# 그래프 설정
ax.set_xlabel('Trial')
ax.set_ylabel('Value')
ax.set_title('VNS vs Sham Data by Trial')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

# 그래프 보여주기
plt.show()


#%%  plot - 2 - trial 두개 합치기

# VNS와 sham 그룹의 평균값과 SEM을 계산 (x축을 3으로 축소)
vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

# trials = mssave[0]
i = [0,1]
for i in [[0,1], [2,3], [4,5]]:
    # trials = np.mean(np.array([mssave[i[0]], mssave[i[1]]]), axis=0) # 평균내기
    
    trials = [np.mean(np.array([mssave[i[0]][0], mssave[i[1]][0]]), axis=0),\
              np.mean(np.array([mssave[i[0]][1], mssave[i[1]][1]]), axis=0)] # 같이 취급하기
    
    vns_means.append(np.mean(trials[0]))
    sham_means.append(np.mean(trials[1]))
    
    vns_sems.append(np.std(trials[0]) / np.sqrt(len(trials[0])))
    print(len(trials[0]), len(trials[1]))
    sham_sems.append(np.std(trials[1]) / np.sqrt(len(trials[1])))

x = np.arange(1, 4)
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)

# 그래프 설정
ax.set_xlabel('Trial')
ax.set_ylabel('Stroop score')
ax.set_title('VNS vs Sham Data by Trial')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

# 그래프 보여주기
plt.show()
for i in [[0,1], [2,3], [4,5]]:
    # trials = np.mean(np.array([mssave[i[0]], mssave[i[1]]]), axis=0) # 평균내기
    
    trials = [np.concatenate((mssave[i[0]][0], mssave[i[1]][0]), axis=0),\
              np.concatenate((mssave[i[0]][1], mssave[i[1]][1]), axis=0)] # 같이 취급하기

    t_stat, p_value = stats.ttest_ind(trials[0], trials[1])
    print(i)
    print("t-statistic:", t_stat)
    print("p-value:", p_value)



#%%  plot - 2 - trial 두개 합치기 - first nmr

# VNS와 sham 그룹의 평균값과 SEM을 계산 (x축을 3으로 축소)
vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

# trials = mssave[0]
i = [0,1]
# trials_nmr = np.mean(np.array([mssave[i[0]], mssave[i[1]]]), axis=0)

vns_nmr = np.mean(np.array([mssave[i[0]][0], mssave[i[1]][0]]), axis=0)
sham_nmr = np.mean(np.array([mssave[i[0]][1], mssave[i[1]][1]]), axis=0)


for i in [[0,1], [2,3], [4,5]]:
    vns = np.mean(np.array([mssave[i[0]][0], mssave[i[1]][0]]), axis=0)
    sham = np.mean(np.array([mssave[i[0]][1], mssave[i[1]][1]]), axis=0)
    
    vns = vns - vns_nmr
    sham = sham  sham_nmr
    
    trials = [vns, sham] 
    vns_means.append(np.mean(trials[0]))
    sham_means.append(np.mean(trials[1]))
    
    vns_sems.append(np.std(trials[0]) / np.sqrt(len(trials[0])))
    sham_sems.append(np.std(trials[1]) / np.sqrt(len(trials[1])))

x = np.arange(1, 4)
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)
bars2 = ax.bar(x + width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)

# 그래프 설정
ax.set_xlabel('Trial')
ax.set_ylabel('Value')
ax.set_title('VNS vs Sham Data by Trial')
ax.set_xticks(x)
ax.set_xticklabels(x)
ax.legend()

# 그래프 보여주기
plt.show()

t_stat, p_value = stats.ttest_ind(trials[0], trials[1])
print(i)
print("t-statistic:", t_stat)
print("p-value:", p_value)




























































































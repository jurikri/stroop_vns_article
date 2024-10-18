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
# import pickle
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

import os
# 파일 경로
file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db.xlsx'

# 파일 삭제
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"File {file_path} has been deleted.")
else:
    print(f"The file {file_path} does not exist.")
           
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

keylist = list(set(msdict_array[:,14]))
mssave = []
for k in range(len(keylist)):
    msid = keylist[k]
    vix = np.where(msdict_array[:,14]==msid)[0]
    tmp = msdict_array[vix[[0,1]]]
    gap = np.array(tmp[:,9], dtype=float) - np.array(tmp[:,6], dtype=float)
    mssave.append([msid, np.mean(gap), tmp[0,-1]])

#%% Figure2 -Figure2A
import stroop_score_calgen
import msFunction


i = 0
tscore_save = []
for i in range(len(msdict_array)):
    data = np.array(msdict_array[i][[3,6,9,2,5,8]], dtype=float)  
    score, _ = stroop_score_calgen.stroop_tscore(data, weights = [[1, 0], [0.33, 0.33, 0.33]])
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
    
#%  plot - 2 - trial 두개 합치기
# VNS와 sham 그룹의 평균값과 SEM을 계산 (x축을 3으로 축소)
vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

# trials = mssave[0]
i = [0,1]
for i in [[0,1], [2,3], [4,5]]:
    trials = [np.mean(np.array([mssave[i[0]][0], mssave[i[1]][0]]), axis=0),\
              np.mean(np.array([mssave[i[0]][1], mssave[i[1]][1]]), axis=0)] # 같이 취급하기
    
    print(len(trials[0]), len(trials[1]))
    vns_means.append(np.median(trials[0]))
    sham_means.append(np.median(trials[1]))
    
    vns_sems.append(np.std(trials[0]) / np.sqrt(len(trials[0])))
    sham_sems.append(np.std(trials[1]) / np.sqrt(len(trials[1])))

fig, ax = plt.subplots(figsize=(4.36/2, 2.45/2))  # Convert cm to inches

# Adjust the position of the plot area to take up most of the figure
ax.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]

# Bar width and x positions
width = 0.35
x = np.array(range(len(sham_means)))

# Plotting bars
bars1 = ax.bar(x - width/2, sham_means, width, yerr=sham_sems, label='Sham', \
               capsize=2, color='#B7FFF4', error_kw=dict(lw=0.6, capthick=0.6))
    
bars2 = ax.bar(x + width/2, vns_means, width, yerr=vns_sems, label='VNS', \
               capsize=2, color='#E2DCFF', error_kw=dict(lw=0.6, capthick=0.6))

    
sham_means_figure3a, sham_sems_figure3a, vns_means_figure3a, vns_sems_figure3a = list(sham_means), list(sham_sems), list(vns_means), list(vns_sems)
    
# ax.set_xlabel('Session #', fontsize=7)
# ax.set_ylabel('Stroop Task\nIntegrated Score', fontsize=7)
# ax.set_title('Reaction Time', fontsize=7)
# ax.set_title('Accuracy', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(x + 1, fontsize=7)
ax.tick_params(width=0.2)  # Adjust the value for the desired thickness of the ticks

ax.set_ylim([-0.3, 1])  # Replace xmin and xmax with your desired range
ax.set_yticks(np.arange(-0.2, 1.1, 0.2))  # Replace start, stop, step with your desired values
ax.tick_params(axis='y', labelsize=7)  # Replace 7 with the desired font size

for spine in ax.spines.values():
    spine.set_linewidth(0.2) 
    
# file_path_corrected = 'figure_with_corrected_dimensions.png'
# plt.savefig(file_path_corrected, dpi=300, bbox_inches='tight')

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure2A.png'
    # plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

for i in [[0,1], [2,3], [4,5]]:
    trials = [np.concatenate((mssave[i[0]][0], mssave[i[1]][0]), axis=0),\
              np.concatenate((mssave[i[0]][1], mssave[i[1]][1]), axis=0)] # 같이 취급하기

    t_stat, p_value = stats.ttest_ind( trials[0], trials[1])
    print(i)
    print("t-statistic:", t_stat)
    print("p-value:", p_value)


#%% Figure2 -Figure2C
import stroop_score_calgen
import msFunction

i = 0
tscore_save = []
for i in range(len(msdict_array)):
    data = np.array(msdict_array[i][[3,6,9,2,5,8]], dtype=float)  
    score, _ = stroop_score_calgen.stroop_tscore(data, weights = [[0, 1], [0.33, 0.33, 0.33]])
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
    
#%  plot - 2 - trial 두개 합치기
# VNS와 sham 그룹의 평균값과 SEM을 계산 (x축을 3으로 축소)
vns_means, sham_means = [], []
vns_sems, sham_sems = [], []

# trials = mssave[0]
i = [0,1]
for i in [[0,1], [2,3], [4,5]]:
    trials = [np.mean(np.array([mssave[i[0]][0], mssave[i[1]][0]]), axis=0),\
              np.mean(np.array([mssave[i[0]][1], mssave[i[1]][1]]), axis=0)] # 같이 취급하기
    
    print(len(trials[0]), len(trials[1]))
    vns_means.append(np.median(trials[0]))
    sham_means.append(np.median(trials[1]))
    
    vns_sems.append(np.std(trials[0]) / np.sqrt(len(trials[0])))
    sham_sems.append(np.std(trials[1]) / np.sqrt(len(trials[1])))
    
sham_means_figure3c, sham_sems_figure3c, vns_means_figure3c, vns_sems_figure3c = list(sham_means), list(sham_sems), list(vns_means), list(vns_sems)


fig, ax = plt.subplots(figsize=(4.36/2, 2.45/2))  # Convert cm to inches

# Adjust the position of the plot area to take up most of the figure
ax.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]

# Bar width and x positions
width = 0.35
x = np.array(range(len(sham_means)))

# Plotting bars
bars1 = ax.bar(x - width/2, sham_means, width, yerr=sham_sems, label='Sham', \
               capsize=2, color='#B7FFF4', error_kw=dict(lw=0.6, capthick=0.6))
    
bars2 = ax.bar(x + width/2, vns_means, width, yerr=vns_sems, label='VNS', \
               capsize=2, color='#E2DCFF', error_kw=dict(lw=0.6, capthick=0.6))

ax.set_xlabel('Session #', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(x + 1, fontsize=7)
ax.tick_params(width=0.2)  # Adjust the value for the desired thickness of the ticks

ax.set_ylim([0, 1])  # Replace xmin and xmax with your desired range
ax.set_yticks(np.arange(0, 1.1, 0.2))  # Replace start, stop, step with your desired values
ax.tick_params(axis='y', labelsize=7)  # Replace 7 with the desired font size

for spine in ax.spines.values():
    spine.set_linewidth(0.2) 
    
# file_path_corrected = 'figure_with_corrected_dimensions.png'
# plt.savefig(file_path_corrected, dpi=300, bbox_inches='tight')

if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure2C.png'
    # plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

for i in [[0,1], [2,3], [4,5]]:
    trials = [np.concatenate((mssave[i[0]][0], mssave[i[1]][0]), axis=0),\
              np.concatenate((mssave[i[0]][1], mssave[i[1]][1]), axis=0)] # 같이 취급하기

    t_stat, p_value = stats.ttest_ind( trials[0], trials[1])
    print(i)
    print("t-statistic:", t_stat)
    print("p-value:", p_value)


#%% 기본정보 변수에 저장 및 export

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
        # trials = [np.mean(np.array([mssave_rt[i[0]][0][fn], mssave_rt[i[1]][0][fn]]), axis=0),\
        #           np.mean(np.array([mssave_rt[i[0]][1][fn], mssave_rt[i[1]][1][fn]]), axis=0)] # 같이 취급하기
            
        trials = [np.concatenate((mssave_rt[i[0]][0][fn], mssave_rt[i[1]][0][fn]), axis=0),\
                  np.concatenate((mssave_rt[i[0]][1][fn], mssave_rt[i[1]][1][fn]), axis=0)] # 같이 취급하기

        msdict_save[dict_name[fn]][str(i)] = {}
        print(len(trials[0]))
        
        # if dict_name[fn] in ['neutral_rt', 'congruent_rt', 'incongruent_rt']:
        msdict_save[dict_name[fn]][str(i)]['VNS'] = trials[0]
        msdict_save[dict_name[fn]][str(i)]['sham'] = trials[1]
        # else:
        #     msdict_save[dict_name[fn]][str(i)]['VNS'] = 1 - trials[0]
        #     msdict_save[dict_name[fn]][str(i)]['sham'] = 1 - trials[1]

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

    
#%% Figure2B
# 2x3 subplot 설정
import seaborn as sns
sessions = [[0, 1], [2, 3], [4, 5]]

fig, axs = plt.subplots(2, 3, figsize=(4.36/2 * 3 * 0.7, 2.45/2 * 2 * 1.4))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
# plt.subplots_adjust(hspace=0.6)  # 기본값은 0.2입니다. 0.6으로 늘려서 간격을 더 넓힙니다.


for ix, n in enumerate(dict_name):
    vns_data, sham_data = [], []
    for se in sessions:
        vns = msdict_save[n][str(se)]['VNS']
        sham = msdict_save[n][str(se)]['sham']
        
        vns_data.extend(vns)
        sham_data.extend(sham)
        
        t_stat, p_value = stats.ttest_ind(sham, vns)
        print()
        print(n, se)
        print("t-statistic:", t_stat)
        print("p-value:", p_value)

    # Combine data for violin plot
    all_data = []
    group_labels = []
    session_labels = []

    for i, se in enumerate(sessions):
        all_data.extend(msdict_save[n][str(se)]['sham'])
        group_labels.extend(['Sham'] * len(msdict_save[n][str(se)]['sham']))
        session_labels.extend([f'{i+1}'] * len(msdict_save[n][str(se)]['sham']))
        
        all_data.extend(msdict_save[n][str(se)]['VNS'])
        group_labels.extend(['VNS'] * len(msdict_save[n][str(se)]['VNS']))
        session_labels.extend([f'{i+1}'] * len(msdict_save[n][str(se)]['VNS']))

    # 현재 subplot에 대한 ax 선택
    ax = axs[ix // 3, ix % 3]
    ax.tick_params(width=0.2)  # Adjust the value for the desired thickness of the ticks
    
    # Define colors
    sham_color = '#B7FFF4'
    vns_color = '#E2DCFF'
    sham_scatter_color = '#7CC7C7'
    vns_scatter_color = '#A794C3'
    sham_violin_line_color = '#B7FFF4'
    vns_violin_line_color = '#E2DCFF'

    # Plotting violin plot with specified colors
    sns.violinplot(x=session_labels, y=all_data, hue=group_labels, split=True, inner=None, ax=ax,
                   palette={'Sham': sham_color, 'VNS': vns_color}, alpha=1, linewidth=0, legend=False)

        
    if ix in [0]: ax.set_ylabel('ms', fontsize=7)
    if ix in [3]: ax.set_ylabel('False ratio', fontsize=7)
        
    if ix in [3,4,5]:
        ax.set_xlabel('Session #', fontsize=7)
    # ax.set_title(n, fontsize=9)
    
    if ix in [0,1,2]:
        ax.set_ylim([300, 1200])  # Replace xmin and xmax with your desired range
        ax.set_yticks(np.arange(300, 1401, 200))
        
    if ix in [3,4,5]:
        ax.set_ylim([-0.2, 0.8])  # Replace xmin and xmax with your desired range
        ax.set_yticks(np.arange(0, 0.9, 0.2)) 
        
    
    # if ix in [0]:
    #     ax.set_ylabel('Reaction Time (ms)', fontsize=7)
    
    # elif ix in [3]:
    #     ax.set_ylabel('False Ratio', fontsize=7)
        
    for ax in axs.flat:
        ax.tick_params(axis='x', labelsize=7)  # x tick 폰트 크기 10으로 설정
        ax.tick_params(axis='y', labelsize=7)  # y tick 폰트 크기 10으로 설정


    # if ix == 0:
    #     ax.legend()
    # else:
    # legend = ax.get_legend()
    # if legend is not None:
    #     legend.remove()

# 레이아웃 조정 및 플롯 출력
# plt.tight_layout()
# plt.show()
for ax in axs.flat:
    for spine in ax.spines.values():
        spine.set_linewidth(0.2)


if True:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure2B.png'
    # plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()
    


#%% Figure1, B,C, 기본정보 bar plot

from scipy import stats
from matplotlib.patches import Patch

dict_name = ['neutral_rt', 'congruent_rt', 'incongruent_rt', \
             'neutral_acc', 'congruent_acc', 'incongruent_acc']
             
figure_title = ['Neutral Cue', 'Congruent Cue', 'Incongruent Cue', \
                'Neutral Cue', 'Congruent Cue', 'Incongruent Cue']
    
sessions = [[0, 1], [2, 3], [4, 5]]
# vns_data, sham_data = [], []

#%%
# for ix, n in enumerate(dict_name):
vns_means, vns_sems, sham_means, sham_sems = [] ,[] ,[] ,[]
# vns_data, sham_data = [], []
for se in sessions:
    
    t1 = 'congruent_rt'
    t2 = 'incongruent_rt'
    
    vns = msdict_save[t2][str(se)]['VNS'] - msdict_save[t1][str(se)]['VNS']
    sham = msdict_save[t2][str(se)]['sham'] - msdict_save[t1][str(se)]['sham']
    
    vns_means.append(np.mean(vns))    
    vns_sems.append(stats.sem(vns))
  
    sham_means.append(np.mean(sham))    
    sham_sems.append(stats.sem(sham))
    
    t_stat, p_value = stats.ttest_ind(sham, vns)
    print(n, se)
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

# data = [sham_data[0], vns_data[0], sham_data[1], vns_data[1], sham_data[2], vns_data[2]]

x = np.arange(1, 4)

# 바의 너비
width = 0.35
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, sham_means, width, yerr=sham_sems, label='Sham', capsize=5)
bars2 = ax.bar(x + width/2, vns_means, width, yerr=vns_sems, label='VNS', capsize=5)

# 그래프 설정
ax.set_xlabel('Session #')

ax.set_title(n)
ax.set_xticks(x)
ax.set_xticklabels(x)

if ix in [0,1,2]:
    # ax.set_ylim([500, 950])
    ax.set_ylabel('reaction time (ms)')

elif ix in [3,4,5]:
    # ax.set_ylim([0, 0.])
    ax.set_ylabel('false ratio')
    
ax.legend()

# 그래프 보여주기
if False:
    fsave = r'C:\\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure'
    fsave2 = fsave + '\\Figure1_basic_' + str(ix) + '.png'
    plt.tight_layout()
    plt.savefig(fsave2, dpi=200 ,bbox_inches='tight', pad_inches=0.1)
    plt.show()

plt.show()

#%%

import pickle


# 필요한 변수를 딕셔너리에 저장
msdict_figure_all = {
    'x': np.arange(1, 4),
    'width': 0.35,
    'sham_means_figure3a': sham_means_figure3a,
    'sham_sems_figure3a': sham_sems_figure3a,
    'vns_means_figure3a': vns_means_figure3a,
    'vns_sems_figure3a': vns_sems_figure3a,
    'sham_means_figure3c': sham_means_figure3c,
    'sham_sems_figure3c': sham_sems_figure3c,
    'vns_means_figure3c': vns_means_figure3c,
    'vns_sems_figure3c': vns_sems_figure3c,
    'dict_name': dict_name,
    'sessions': sessions,
    'msdict_save': msdict_save
}

# 저장 경로 설정
spath = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\result-figure' + '\\'

# pickle 파일로 저장
with open(spath + 'vnstroop_figure2.pickle', 'wb') as file:
    pickle.dump(msdict_figure_all, file)

print("모든 변수들이 성공적으로 저장되었습니다.")




















































































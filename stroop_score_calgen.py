# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:11:41 2024

@author: PC
"""

if False:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import pickle
    
    # Define the correct file path
    file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db.xlsx'
    df = pd.read_excel(file_path)
    
    
    #%% session 번호 부여
    
    # 'msid2' 컬럼을 '_'를 기준으로 분리하여 'date'와 'name' 컬럼을 생성
    df[['date', 'name']] = df['msid2'].str.split('_', n=1, expand=True)
    df['date'] = df['msid2'].str[:8]
    
    # 각 'date'와 'name' 조합별 데이터셋의 행 수를 확인
    dataset_counts = df.groupby(['date', 'name']).size()
    
    # DataFrame을 'date', 'name', 'msid2' 순으로 정렬
    df_filtered = df.sort_values(by=['date', 'name', 'msid2'])
    
    # 각 'date'와 'name' 조합 내에서 세션 번호를 할당
    df_filtered['session'] = df_filtered.groupby(['date', 'name']).cumcount()
    
    # 새로운 DataFrame df2 생성
    df2 = df_filtered.copy()
    df2.head()  # 첫 몇 개의 행을 표시하여 확인
    df2['ID'] = df2['date'] + '_' + df2['name']
    
    pickle_file_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl\stroop_score_db.pkl'
    df2.to_pickle(pickle_file_path)
    
    #%% vns_stroop_data
    
    df3 = pd.read_pickle(pickle_file_path)
    df3_array = np.array(df3)
    session_n_list = np.array(df3_array[:,13], dtype=int)
    session_n_list = np.array(df3_array[:,13], dtype=int)
    s12 = np.where(np.logical_or((session_n_list == 0), (session_n_list == 1)))[0]
    vns_stroop_df = df3_array[s12]
    
    #%% WIS (정보학회)
    eload = r"E:\EEGsaves\stroop_0423_WIS 박람회\stroop_score_save_2024WIS박람회\stroop_score_db.xlsx"
    WIS_df = pd.read_excel(eload)
    
    #%% WBC (생체재료 학회)
    eload = r"E:\EEGsaves\stroop_0612_WBC 학회\stroop_score\stroop_score_db.xlsx"
    WBC_df = pd.read_excel(eload)
    
    #%%
    
    merge_df = np.concatenate((vns_stroop_df[:,:11], np.array(WIS_df), np.array(WBC_df)), axis=0)
    
    #%%
    
    import numpy as np
    import pandas as pd
    
    # z-score 계산 함수
    def calculate_zscore(data, mean, std):
        return (data - mean) / std
    
    def normalize_to_100_scale(x, x1, x2):
        return 100 * (x - x2) / (x1 - x2)
    
    # 필요한 컬럼 추출
    rt_columns = np.array(merge_df[:, [3, 6, 9]], dtype=float)
    acc_columns = np.array(merge_df[:, [2, 5, 8]], dtype=float)
    
    columns = [rt_columns[:, 0], rt_columns[:, 1], rt_columns[:, 2], 
               acc_columns[:, 0], acc_columns[:, 1], acc_columns[:, 2]]
    
    z_scores = []
    means_stds = []
    for ix, col in enumerate(columns):
        mean = np.nanmean(col)
        std = np.nanstd(col)
        means_stds.append((mean, std))
        z_score = calculate_zscore(col, mean, std)
        if ix in [0,1,2]: z_score = -z_score
        z_scores.append(z_score)
        
    means_stds_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\means_stds.pkl'
    with open(means_stds_path, 'wb') as f:
        pickle.dump(means_stds, f)
        
    predatsets = stroop_tscore(columns, weights = [[0.5, 0.5], [0.25, 0.25, 0.5]])
    
    # np.nanmean(predatsets[1][0:3])
    # np.nanmean(predatsets[1][3:6])
    
    smin = np.nanmin(predatsets[0])
    smax = np.nanmax(predatsets[0])
    
def stroop_tscore(data, weights = [[0.8, 0.2], [0.2, 0.3, 0.5]]):
    
    def min_max_scale(data, feature_range=(-0.4, 1.07)):
        data_min, data_max = feature_range
        scaled_data = (data - data_min) / (data_max - data_min)

        return scaled_data
    
    import pickle
    import numpy as np
    def calculate_zscore(data, mean, std):
        return (data - mean) / std

    means_stds_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\means_stds.pkl'
    with open(means_stds_path, 'rb') as f:
        loaded_means_stds = pickle.load(f)
        
    # minmax_path = r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\lognorm_parameters.pkl'
    # with open(minmax_path, 'rb') as f:
    #     minmax = pickle.load(f)
    
    z_scores = []
    for ix in range(6):
        mean = loaded_means_stds[ix][0]
        std = loaded_means_stds[ix][1]

        z_score = calculate_zscore(data[ix], mean, std)
        if ix in [0,1,2]: z_score = -z_score
        z_scores.append(z_score)
    
    z_scores = np.array(z_scores)
 
    fscore_rt = ((z_scores[0] * weights[1][0]) + (z_scores[1] * weights[1][1]) + (z_scores[2] * weights[1][2])) * weights[0][0]
    fscore_acc = ((z_scores[3] * weights[1][0]) + (z_scores[4] * weights[1][1]) + (z_scores[5] * weights[1][2])) * weights[0][1]  
    fscore = fscore_rt + fscore_acc
    
    return min_max_scale(fscore), z_scores
        
#%%
if False:
    z_scores2 = np.array(z_scores)
    ssave = []
    for i in range(z_scores2.shape[1]):
        ssave.append(stroop_tscore(z_scores2[:,i]))
    
    
    

        
        
    





# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:09:36 2024

@author: PC
"""
# psave_path = r'E:\EEGsaves\stroop_0423_WIS 박람회\stroop_score_save_2024WIS박람회'
def msmain(block_path=None, psave_path=None):
    import os
    import glob
    import pickle
    import numpy as np
    import pandas as pd
    
    # 모든 pkl listup 하기
    pattern = os.path.join(block_path, "*.pkl")
    pkl_files = glob.glob(pattern)
    
    # session ID 검출하기
    path_parts = block_path.split(os.sep)
    directory_variable = None
    # Loop through path parts to find the 'block' directory and get the preceding directory name
    for i, part in enumerate(path_parts):
        if part.startswith('block'):
            if i > 0:
                directory_variable = path_parts[i - 1]
                break
            else:
                print("No preceding directory exists before 'block'.")
                break
    print("Directory name before 'block':", directory_variable)

    
    if len(pkl_files) == 60:
        # _, msid = directory_variable.rsplit('_', 1)
        # date_text = directory_variable.split('_', 1)[0]  # 언더바 기준으로 분할하여 좌측 부분(날짜와 시간)을 가져옴
        # data = date_text[4:8] + date_text[8:12]      
        # msid2 = msid + '_' + data
        # parent_dir = os.path.dirname(block_path)  # block2 폴더 경로
        # grandparent_dir = os.path.dirname(parent_dir)  # 202405261555_LJH 폴더 경로
        
        # block2와 202405261555_LJH 폴더 이름 추출
        grandparent_dir = os.path.dirname(block_path)
        block_folder = os.path.basename(block_path)
        experiment_folder = os.path.basename(grandparent_dir)
        
        print("Block Folder:", block_folder)
        print("Experiment Folder:", experiment_folder)
        
        date_part, identifier = experiment_folder.split('_', 1)
        # date_part = date_part[:8]
        print("Date Part:", date_part)
        print("Identifier:", identifier)
        msid = date_part + '_' + identifier
        
        block_n = 0
        mssave_array = np.zeros((1, 3, 3)) * np.nan # block x group x fn
        mssave_dict = {}; blcok_name = block_folder
        mssave_dict[blcok_name] = {} 
        accuracy = {'neutral':[], 'congruent':[], 'incongruent':[]}
        for j in range(len(pkl_files)):
            with open(pkl_files[j], 'rb') as file:
                msdict = pickle.load(file)
            
            label = None
            if msdict['Correct'] and msdict['Response']=='Yes': label = 'TP'
            elif msdict['Correct'] and msdict['Response']=='No': label = 'FP'
            elif not(msdict['Correct']) and msdict['Response']=='Yes': label = 'FN'
            elif not(msdict['Correct']) and msdict['Response']=='No': label = 'TN'
            if msdict['Response Time'] == 1500: label = 'Noresponse'
    
            accuracy[msdict['Condition']].append([label, msdict['Response Time']])
            
            if msdict['Response Time'] < 200: label = 'FP' # 200 ms 이하는 오입력으로 판정함

        keylist = list(accuracy.keys())
        for c in range(len(keylist)):
            gn = keylist[c]
            # rt_save[str(t)][gn] = []
            
            TP_vix = np.where(np.array(accuracy[gn])[:,0] == 'TP')[0]
            TP_n = len(TP_vix)
            TP_rts = np.array(np.array(accuracy[gn])[TP_vix,1], dtype=float)
    
            FP_vix = np.where(np.array(accuracy[gn])[:,0] == 'FP')[0]
            FP_n = len(FP_vix)
            FP_rts = np.array(np.array(accuracy[gn])[FP_vix,1], dtype=float)
            
            TN_vix = np.where(np.array(accuracy[gn])[:,0] == 'TN')[0]
            TN_n = len(TN_vix)
            TN_rts = np.array(np.array(accuracy[gn])[TN_vix,1], dtype=float)
            
            FN_vix = np.where(np.array(accuracy[gn])[:,0] == 'FN')[0]
            FN_n = len(FN_vix)
            FN_rts = np.array(np.array(accuracy[gn])[FN_vix,1], dtype=float)
            
            if (TP_n + TN_n + FP_n + FN_n) != 0:
                acc = (TP_n + TN_n) / (TP_n + TN_n + FP_n + FN_n)
            else: acc = 0
            
            T_rt = np.nanmean(np.concatenate((TP_rts, TN_rts), axis=0))
            F_rt = np.nanmean(np.concatenate((FP_rts, FN_rts), axis=0))
            # nan 왜찍힘?
            # print(t, n, keylist[c], acc, np.round([T_rt, F_rt]))
            # rt_save[str(t)][keylist[c]] = [T_rt, F_rt, acc]
            
            mssave_dict[blcok_name][gn] = {}
            mssave_dict[blcok_name][gn]['acc'] = acc
            mssave_dict[blcok_name][gn]['true_reatction_time'] = T_rt
            mssave_dict[blcok_name][gn]['false_reatction_time'] = F_rt
            
            mssave_array[block_n, c, 0] = acc
            mssave_array[block_n, c, 1] = T_rt
            mssave_array[block_n, c, 2] = F_rt
            
        mssave_array[0,:,0][np.isnan(mssave_array[0,:,0])] = 0
        mssave_array[0,:,1:][np.isnan(mssave_array[0,:,1:])] = 2000
            
        participant_data = np.reshape(mssave_array[0], (9,))
  
        def metric_stroop(participant_data):
             # 가중치 설정
             accuracy_weights = np.array([100, 100, 150])  # 정확도에 대한 가중치
             response_time_weights = np.array([1, 1, 1.5])  # 반응 시간에 대한 가중치
             wrong_response_penalty = 1.2  # 오답에 대한 추가 패널티
             # 정확도와 반응 시간을 별도로 계산
             accuracy_scores = participant_data[[0, 3, 6]] * accuracy_weights
             response_time_scores = 1 / \
                 (participant_data[[1, 4, 7]] + participant_data[[2, 5, 8]] * \
                  wrong_response_penalty) * response_time_weights
             # 최종 점수 계산: 정확도 점수와 반응 시간 점수의 합산
             final_score = (np.sum(accuracy_scores) * 3) + (np.sum(response_time_scores) * 1e5)
             final_score = np.round(final_score / 10, 3) - 54
             print(f"Final score: {final_score}")
             return final_score
         
        final_score = metric_stroop(participant_data)
        
        mssave_dict['final_score_' + blcok_name] = final_score
        mssave_dict['msid2'] = msid
        
        file_path = psave_path + '\\' + msid + '_' + block_folder + '.pkl'
        # Save the dictionary to a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(mssave_dict, file)
        print("Dictionary has been saved to", file_path)
        
    
        
        
        

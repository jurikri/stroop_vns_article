# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:19:36 2024

@author: PC
"""


"""
list up 후 ranking_board_backend 이용해서 계산, pkl로 저장
"""

# import os
# import sys
# sys.path.append(r'E:\EEGsaves\stroop_0423_WIS 박람회')
# import glob
# import pickle
# import ranking_board_backend

# # Define the root path where the folders are located
# root_path = r'E:\EEGsaves\stroop_0423_WIS 박람회\Stroop_score'
# psave_path = r'E:\EEGsaves\stroop_0423_WIS 박람회\stroop_score_save_2024WIS박람회'

# # List all directories in the root path
# directories = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
# print("Directories found:", directories)

# # List all .pkl files within the 'block1' subdirectory of each directory
# directory = directories[0]
# for directory in directories:
#     block1_path = os.path.join(root_path, directory, 'block1')
#     ranking_board_backend.msmain(block1_path=block1_path, psave_path=psave_path)

"""
ranking_board_backend *.pkl 로드, DB 형태로 저장
[ID, neutral-(acc,rt1,rt2), conguent-(acc,rt1,rt2), incongruent-(acc,rt1,rt2), score]
"""
def msmain(directory_path=r'C:\SynologyDrive\worik in progress\혁창패\임상실험 논문작성\analysis\stroop_score_pkl'):
    import pandas as pd
    import os
    import glob
    import pickle
    
    # List all .pkl files in the directory
    pkl_files = glob.glob(os.path.join(directory_path, '*.pkl'))
    
    # Prepare a list to store each row of data
    data_rows = []
    
    # Load each .pkl file and extract data
    file_path = pkl_files[0]
    for file_path in pkl_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        # Extract data for each file and create a row
        msid = data['msid2']
        block = list(data.keys())[0]
        neutral = data[block]['neutral']
        congruent = data[block]['congruent']
        incongruent = data[block]['incongruent']
    
        # Create a row with the required format
        row = [
            msid, block,
            neutral['acc'], neutral['true_reatction_time'], neutral['false_reatction_time'],
            congruent['acc'], congruent['true_reatction_time'], congruent['false_reatction_time'],
            incongruent['acc'], incongruent['true_reatction_time'], incongruent['false_reatction_time']
        ]
        data_rows.append(row)
    
    # Create a DataFrame with the specified columns
    columns = [
        'msid2', 'blocknum', 
        'neutral-acc', 'neutral-true_rt', 'neutral-false_rt',
        'congruent-acc', 'congruent-true_rt', 'congruent-false_rt',
        'incongruent-acc', 'incongruent-true_rt', 'incongruent-false_rt'
    ]
    data_frame = pd.DataFrame(data_rows, columns=columns)
    
    # Save the DataFrame to an Excel file
    excel_path = os.path.join(directory_path, 'stroop_score_db.xlsx')
    data_frame.to_excel(excel_path, index=False)
    
    print("Data has been saved to Excel successfully at:", excel_path)


        
            
        






























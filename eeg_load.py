

# 메인 함수
def main(directory_path):
    import glob
    import os
    import pickle
    import numpy as np
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    # 개별 파일을 로드하는 함수
    def load_pkl_file(file_path):
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        except:
            print(file_path)
        
    # .pkl 파일 목록을 가져옵니다
    pkl_files = glob.glob(os.path.join(directory_path, '*.pkl'))

    # 병렬 처리를 위해 ThreadPoolExecutor 사용
    with ThreadPoolExecutor() as executor:
        # 모든 파일을 병렬로 로드합니다
        results = list(tqdm(executor.map(load_pkl_file, pkl_files), total=len(pkl_files)))

    # 결과를 numpy 배열로 병합합니다
    # eeg_df = np.array(results[0])
    # for msdict in results[1:]:
    #     eeg_df = np.concatenate((eeg_df, np.array(msdict)), axis=0)

    # print()
    # print(eeg_df.shape)
    
    eeg_arrays = [np.array(msdict) for msdict in results]
    eeg_df = np.vstack(eeg_arrays)
    
    print(eeg_df.shape)
        
    return eeg_df


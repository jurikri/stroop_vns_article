# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 17:19:26 2023

@author: PC
"""

def msmain(SR=None, ppg_data=None, peak_times=None):
    import numpy as np
    
    if peak_times is None:
        ws1 = int(SR * (60/30)) # 느릴때, 30 bpm
        ws2 = int(SR * (60/100)) # 빠를때 100 bpm
        
        bins = 1
        msbins1 = np.arange(0, len(ppg_data)-ws1, bins, dtype=int)
        msbins2 = np.arange(0, len(ppg_data)-ws2, bins, dtype=int)
        
        ixmatrix = np.zeros((ppg_data.shape[0], 2))
        for w1 in range(len(msbins1)):
            wn1 = ppg_data[msbins1[w1]:msbins1[w1]+ws1]
            ixmatrix[np.argmax(wn1) + msbins1[w1], 0] += 1
            
        for w2 in range(len(msbins2)):
            wn2 = ppg_data[msbins2[w2]:msbins2[w2]+ws2]
            ixmatrix[np.argmax(wn2) + msbins2[w2], 1] += 1
            
        t, d = 100, 10
        # plt.plot(ppg_data[SR*t:SR*(t+d)])
        # plt.plot(ixmatrix[:,0][SR*t:SR*(t+d)])
        # plt.plot(ixmatrix[:,1][SR*t:SR*(t+d)])
        
        peaks = np.where(ixmatrix[:,1] > 50)[0]
        
        if False:
            t, i = 10, 0
            for i in range(0, int(ppg_data.shape[0]/SR), t):
                s = SR * i
                e = SR * (i + t) + (SR*0)
                slice_to_plot = ppg_data[s:e]
                # pix =  np.where((np.sum(accumulated_timeline, axis=0) > 0)[s:e] == 1)[0] + n
                pix = peaks[np.logical_and(peaks>s, peaks<e)]
                pix = pix - s
                time_axis = np.linspace(0, t, len(slice_to_plot))
                
                plt.figure()
                plt.title(str(i))
                plt.plot(time_axis, slice_to_plot)
                peak_times = time_axis[pix]
                plt.scatter(peak_times, slice_to_plot[pix], color='red', label='Detected Peaks', s=10)

        peak_times = peaks # 단위 ms로
        
    peak_times_tw = peak_times
    # def peak_times_in(peak_times_tw):
    # 앞 2번째 부터, 뒤 2번째 peak 까지 잘라서 유효 범위 설정
    # 유효 범위 내에 peak들을 total map에 check,
    # total map에서 2번 이상 중복 체크된것만 유효값으로 check.
    
    # peak 간의 거리 값으로 환산
    # peak time point, 그 peak로 부터 다음 peak로의 거리로 저장
    
    # 거리의 std
    
    # peak_times는 감지된 peak들의 시간 좌표를 나타냅니다.
    NN_intervals = np.diff(peak_times_tw)  # 연속된 peak들 사이의 시간 차이 계산
    SDNN = np.std(NN_intervals) 
    
    # exn = int(len(NN_intervals)*0.01)
    # if exn != 0: NN_intervals_SDNN = np.sort(NN_intervals)[exn:-exn]
    # else: NN_intervals_SDNN = NN_intervals
    # SDNN = np.std(NN_intervals_SDNN)  # NN 간격의 표준 편차 계산
    
    # # RMSSD 계산
    
    NN_intervals_diff = np.diff(NN_intervals)
    RMSSD = np.sqrt(np.mean(np.square(NN_intervals_diff)))
    
    # exn = int(len(NN_intervals_diff)*0.01)
    # NN_intervals_diff_ex = np.sort(NN_intervals_diff)[exn:-exn]
    
    # RMSSD = np.sqrt(np.mean(np.square(NN_intervals_diff_ex)))
    # 부교감신경계가 활성화될 때, 심박수는 빠르게 변화합니다 (예를 들어, 휴식 시 빠르게 감소). 이러한 빠른 변화는 연속된 NN 간격의 차이가 크게 되며, 이는 RMSSD 값이 증가하게 합니다.
    # 반면, 교감신경계의 영향은 보통 더 장기적이며, 연속된 NN 간격의 차이에 덜 민감하게 반응합니다.

    
    # # pNN50 계산
    differences = np.abs(NN_intervals_diff)
    pNN50 = np.sum(differences > 50) / len(differences) * 100  # 50ms는 0.05초에 해당
    # 네, 말씀하신 대로 pNN50을 짧은 시간 윈도우의 관점에서 해석하면, 실제로 심장이 상대적으로 느리게 뛰는 순간들을 카운팅하는 것과 유사하게 볼 수 있습니다. pNN50 지표는 연속된 심박 간의 시간 차이가 50 밀리초 이상인 경우를 카운트하며, 이러한 큰 시간 차이는 심박수가 감소하는 순간을 반영할 수 있습니다.

    # # RRHRV 계산 
    # RR_intervals = np.diff(peak_times)  # RR 간격 계산
    # weighted_diffs = np.diff(RR_intervals) / ((RR_intervals[:-1] + RR_intervals[1:]) / 2)  # 연속적인 RR 간격의 차이를 평균으로 가중치 적용
    # rrHRV = np.mean(weighted_diffs)  # 가중치가 적용된 차이들의 평균
    # RRHRV는 연속적인 RR 간격의 차이를 그들의 평균으로 가중치를 두어 계산하는 방식을 기반으로 합니다. 이 방법은 특히 짧은 RR 시퀀스에 적합합니다

    BPM = (len(peak_times_tw) - 1) / ((peak_times_tw[-1] - peak_times_tw[0])/SR) * 60
    
    if False:
        print('SDNN', np.round(SDNN, 2), 'RMSSD', np.round(RMSSD, 2), 'pNN50', np.round(pNN50, 2)) #, 'RRHRV 준비중..')
        print('BPM', BPM)
    
    return SDNN, RMSSD, pNN50, BPM, peak_times
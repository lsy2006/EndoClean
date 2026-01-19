import pandas as pd
import ast
import numpy as np


def filt(data):
    mean = np.mean(data)
    std_dev = np.std(data)

    lower_bound = mean - 2 * std_dev
    upper_bound = mean + 2 * std_dev

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

def find_split(pre_seq):
    sum = [0, 0, 0, 0, 0, 0]
    for i in range(len(pre_seq)):
        if pre_seq[i] > 5:
            pre_seq[i] = 2
        sum[pre_seq[i]] += 1
    pos0 = []
    pos1 = []
    pos2 = []
    pos3 = []
    pos4 = []
    pos5 = []
    start_points = []
    for i in range(len(pre_seq)):
        exec(f'pos{pre_seq[i]}.append(i)')

    for i in range(0, 6):
        exec(f'pos{i} = filt(pos{i})')

    for i in range(0, 6):
        try:
            exec(f'start_points.append(pos{i}[round(len(pos{i})/2)])')
        except:
            if i != 0:
                exec(f'start_points.append(pos{i-1}[round(len(pos{i-1})/2)]+1)')
            else:
                exec(f'start_points.append(0)')

    start_points[0] = 0
    start_points[-1] = len(pre_seq)

    '''
    cnt = 0
    show_seq = []
    for i in range(len(pre_seq)):
        if cnt < 6:
            if i > start_points[cnt]:
                cnt += 1
        show_seq.append(cnt)
    print(start_points)
    
    return show_seq
    '''

    biggest_rate = 0
    split_points = []
    for i in range(len(start_points)-1):
        final_mid = 0
        biggest_rate = 0
        for mid in range(start_points[i]+1, start_points[i+1] - 1):
            left = 0
            right = 0
            left_tot = 0
            right_tot = 0
            for x in range(start_points[i], mid):
                if pre_seq[x] == i:
                    left += 1
                if pre_seq[x] == i or pre_seq[x] == i+1:
                    left_tot += 1
            for x in range(mid+1, start_points[i+1]):
                if pre_seq[x] == i+1:
                    right += 1
                if pre_seq[x] == i or pre_seq[x] == i+1:
                    right_tot += 1
            try:
                rate = left / left_tot + right / right_tot - abs(left / left_tot - right / right_tot) - abs(left - right) / (left_tot + right_tot) * 0.8
            except ZeroDivisionError:
                continue
            if rate > biggest_rate:
                biggest_rate = rate
                final_mid = mid
        split_points.append(final_mid)
    # print(split_points)
    '''
    cnt = 0
    final_seq = []
    for i in range(len(pre_seq)):
        if cnt < 5:
            if i > split_points[cnt]:
                cnt += 1
        final_seq.append(cnt)
    '''
    return split_points[1], split_points[2]# , final_seq

if __name__ == "__main__":
    for selected_video in ['CJ001-1', 'CJ002-2']:
        df = pd.read_csv('./documents/seg_seq.csv').set_index('video')
        pre_seq = ast.literal_eval(df.loc[selected_video]['pred_seq'])
        BBPS_seq = ast.literal_eval(df.loc[selected_video]['BBPS'])
        frame1, frame2 = find_split(pre_seq)
        print(frame1, frame2)
        tot_BBPS_left = 0
        tot_BBPS_mid = 0
        tot_BBPS_right = 0
        num_BBPS_left = 0
        num_BBPS_mid = 0
        num_BBPS_right = 0
        for now_frame in range(len(pre_seq)):
            now_BBPS = BBPS_seq[now_frame]
            if now_BBPS >= 0:
                if now_frame < frame1:
                    tot_BBPS_left += now_BBPS
                    num_BBPS_left += 1
                elif now_frame < frame2:
                    tot_BBPS_mid += now_BBPS
                    num_BBPS_mid += 1
                else:
                    tot_BBPS_right += now_BBPS
                    num_BBPS_right += 1
        BBPS_left = round(tot_BBPS_left / num_BBPS_left)
        BBPS_mid = round(tot_BBPS_mid / num_BBPS_mid)
        BBPS_right = round(tot_BBPS_right / num_BBPS_right)
        print(BBPS_left, BBPS_mid, BBPS_right)

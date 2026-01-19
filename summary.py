import moviepy.editor as mp
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
    return split_points[1], split_points[2]

def summary(selected_video, root_dir):
    df = pd.read_csv('./documents/seg_seq.csv').set_index('video')
    start_time_df = pd.read_csv('./documents/start_time.csv').set_index('video')
    results_df = pd.read_csv('./documents/results.csv')
    if selected_video in results_df['video'].values:
        return 'The video has been processed!'
    
    pre_seq = ast.literal_eval(df.loc[selected_video]['pred_seq'])
    BBPS_seq = ast.literal_eval(df.loc[selected_video]['BBPS'])
    frame1, frame2 = find_split(pre_seq)
    try:
        video = mp.VideoFileClip(f"{root_dir}/{selected_video.split('-')[0]}/{selected_video.split('-')[1]}.mp4")
    except:
        video = mp.VideoFileClip(f"{root_dir}/{selected_video.split('-')[0]}/{selected_video.split('-')[1]}.MP4")
    fps = round(video.fps)
    start_frame = start_time_df.loc[selected_video]['st_time'] * fps

    total_time = (video.reader.nframes - start_frame) // fps
    left_start_time = start_frame // fps
    left_mid_time = (start_frame + frame1) // fps
    mid_right_time = (start_frame + frame2) // fps
    right_end_time = video.reader.nframes // fps
    
    tot_BBPS_left = 0
    tot_BBPS_mid = 0
    tot_BBPS_right = 0
    num_BBPS_left = 0
    num_BBPS_mid = 0
    num_BBPS_right = 0
    rate_BBPS_left = [0, 0, 0, 0]
    rate_BBPS_mid = [0, 0, 0, 0]
    rate_BBPS_right = [0, 0, 0, 0]
    for now_frame in range(len(pre_seq)):
        now_BBPS = BBPS_seq[now_frame]
        if now_BBPS >= 0:
            if now_frame < frame1:
                tot_BBPS_left += now_BBPS
                num_BBPS_left += 1
                rate_BBPS_left[now_BBPS] += 1
            elif now_frame < frame2:
                tot_BBPS_mid += now_BBPS
                num_BBPS_mid += 1
                rate_BBPS_mid[now_BBPS] += 1
            else:
                tot_BBPS_right += now_BBPS
                num_BBPS_right += 1
                rate_BBPS_right[now_BBPS] += 1
    for i in range(4):
        rate_BBPS_left[i] = rate_BBPS_left[i] / num_BBPS_left
        rate_BBPS_mid[i] = rate_BBPS_mid[i] / num_BBPS_mid
        rate_BBPS_right[i] = rate_BBPS_right[i] / num_BBPS_right
    
    if rate_BBPS_left[0] > 0.1:
        BBPS_left = 0
    else:
        if rate_BBPS_left[1] > 0.20:
            BBPS_left = 1
        else:
            BBPS_left = round(tot_BBPS_left / num_BBPS_left)

    if rate_BBPS_mid[0] > 0.1:
        BBPS_mid = 0
    else:
        if rate_BBPS_left[1] > 0.20:
            BBPS_mid = 1
        else:
            BBPS_mid = round(tot_BBPS_mid / num_BBPS_mid)
    
    if rate_BBPS_right[0] > 0.1:
        BBPS_right = 0
    else:
        if rate_BBPS_right[1] > 0.20:
            BBPS_right = 1
        else:
            BBPS_right = round(tot_BBPS_right / num_BBPS_right)

    print(f'{selected_video}:{BBPS_left}+{BBPS_mid}+{BBPS_right}={BBPS_left+BBPS_mid+BBPS_right}')
    results_new_row = pd.DataFrame({'video': [selected_video], 'total_time': [total_time], 'BBPS_left':[BBPS_left], 'BBPS_mid': [BBPS_mid],
                                 'BBPS_right': [BBPS_right], 'left_start_time': [left_start_time], 'left_end_time': [left_mid_time], 
                                 'mid_start_time': [left_mid_time], 'mid_end_time': [mid_right_time], 'right_start_time': [mid_right_time],
                                 'right_end_time': [right_end_time]})
    results_df = pd.concat([results_df, results_new_row], ignore_index=True)
    results_df.to_csv('./documents/results.csv', index=False)
    return 'Done!'

if __name__ == "__main__":
    for selected_video in ['CJ001-1', 'CJ002-2']:
        summary(selected_video)
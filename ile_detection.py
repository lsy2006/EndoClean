import moviepy.editor as mp
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import torch.nn as nn
from collections import Counter
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx_map = {'ileocecus' : 0, 'ascending_colon' : 1, 'hepatic_flexure' : 6, 'transverse_colon' :2, 'splenic_flexure' : 7, 'descending_colon' : 3, 'sigmoid_colon' : 4, 'rectum' : 5, 'in' : 100}
rmap = {'0' : 'ileocecus', '1' : 'ascending_colon', '2' : 'transverse_colon', '3' : 'descending_colon', '4' : 'sigmoid_colon', '5' : 'rectum'}
h, w = 128, 128
VJmodel_name = 'VJacc9462'
ISmodel_name = 'ISacc4966'
vedio_list = ['CJ001-1', 'CJ002-2', 'CJ003-2', 'CJ004-1']
VJmodel_path = f'./models/{VJmodel_name}.pth'
ISmodel_path = f'./models/{ISmodel_name}.pth'


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize((0.5566, 0.3259, 0.2206), (0.2159, 0.1844, 0.1337))
        ])

# model = MultiModalModel(num_classes=8)
classification_model = models.resnet18(pretrained=False)
classification_model.fc = nn.Linear(classification_model.fc.in_features, 6)
classification_model.load_state_dict(torch.load(ISmodel_path, weights_only=True), strict=False)
classification_model = classification_model.to(device)
classification_model.eval()

VJmodel = models.resnet18(pretrained=False)
VJmodel.fc = nn.Linear(VJmodel.fc.in_features, 2)
VJmodel.load_state_dict(torch.load(VJmodel_path, weights_only=True), strict=False)
VJmodel = VJmodel.to(device)

VJmodel.eval()


def ile_dect(selected_vedio, root_dir):
    st_time_df = pd.read_csv('./documents/start_time.csv')
    if selected_vedio in st_time_df['video'].values:
        return 'The video has been processed!'
    try:
        video = mp.VideoFileClip(f"{root_dir}/{selected_vedio.split('-')[0]}/{selected_vedio.split('-')[1]}.mp4")
    except:
        video = mp.VideoFileClip(f"{root_dir}/{selected_vedio.split('-')[0]}/{selected_vedio.split('-')[1]}.MP4")
    fps = round(video.fps)
    
    now_frame = 0
    now_second = 0
    tot_frame = 0
    pre_ile = []

    ile_fram_list = []
    progress_bar = tqdm(video.iter_frames(), desc=f"{selected_vedio}", total=video.reader.nframes)
    for frame in progress_bar:
        now_frame += 1
        if now_frame % fps == 0:
            now_second += 1

        tot_frame += 1
        image = frame[39:1044, 700:1863]

        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            input_tensor= input_tensor.to(device)
            VJoutputs = VJmodel(input_tensor)
            _, VJpredicted = torch.max(VJoutputs, 1)
    
        now_pred = 0
        if VJpredicted.tolist()[0] == 0:
            pre_ile.append(100)
        else:
            with torch.no_grad():
                input_tensor= input_tensor.to(device)
                outputs = classification_model(input_tensor)
                _, predicted = torch.max(outputs, 1)
            now_pred = predicted.tolist()[0]
            if now_pred == 0:
                ile_fram_list.append(now_frame)
                pre_ile.append(now_pred)
            else:
                pre_ile.append(100)
        # print(now_second, rmap[str(pred)], rmap[str(now_pred)], target)
    
    ax = plt.boxplot(ile_fram_list)
    st_frame, ed_frame = round(ax['whiskers'][0].get_ydata()[0]), round(ax['whiskers'][1].get_ydata()[-1])
    # print(st_frame, ed_frame)
    st_time = st_frame // fps
    ed_time = ed_frame // fps
    time_new_row = pd.DataFrame({'video': [selected_vedio], 'st_time': [st_time], 'ed_time': [ed_time]})
    st_time_df = pd.concat([st_time_df, time_new_row], ignore_index=True)
    st_time_df.to_csv('./documents/start_time.csv', index=False)
    return 'Done!'


if __name__ == '__main__':
    for selected_vedio in vedio_list:
        print(selected_vedio, ':')
        st_time_df = pd.read_csv('./documents/start_time.csv')
        st_frame, ed_frame, st_time, ed_time = ile_dect(selected_vedio)
        print(st_frame, ed_frame, st_time, ed_time)
        time_new_row = pd.DataFrame({'video': [selected_vedio], 'st_time': [st_time], 'ed_time': [ed_time]})
        st_time_df = pd.concat([st_time_df, time_new_row], ignore_index=True)
        st_time_df.to_csv('./documents/start_time.csv', index=False)
        print()

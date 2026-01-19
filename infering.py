import moviepy.editor as mp
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
idx_map = {'ileocecus' : 0, 'ascending_colon' : 1, 'hepatic_flexure' : 6, 'transverse_colon' :2, 'splenic_flexure' : 7, 'descending_colon' : 3, 'sigmoid_colon' : 4, 'rectum' : 5}
rmap = {'0' : 'ileocecus', '1' : 'ascending_colon', '2' : 'transverse_colon', '3' : 'descending_colon', '4' : 'sigmoid_colon', '5' : 'rectum'}
#colonmapper_map = {'ileocecus' : 0, 'ascending' : 1, 'hepatic' : 6, 'transverse' : 2, 'splenic_flexure' : 7, 'descending' : 3, 'sigmoid' : 4, 'rectum' : 5, 'N' : 100}
colonmapper_map = {'ileocecus' : 0, 'ascending_colon' : 1, 'hepatic_flexure' : 6, 'transverse_colon' : 2, 'splenic_flexure' : 7, 'descending_colon' : 3, 'sigmoid_colon' : 4, 'rectum' : 5, 'N' : 100}
h = 128
w = 128
VJmodel_name = 'VJacc9462'
ISmodel_name = 'ISacc4966'
BBPS_model_name = 'BBPS_90'
selected_video = ['CJ001-1', 'CJ002-2', 'CJ003-2', 'CJ004-1']
VJmodel_path = f'./models/{VJmodel_name}.pth'
ISmodel_path = f'./models/{ISmodel_name}.pth'
BBPSmodel_path = f'./models/{BBPS_model_name}.pth'

BBPS_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize((0.5566, 0.3259, 0.2206), (0.2159, 0.1844, 0.1337))
])

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

class Classifier(nn.Module):
    def __init__(self, num_classes=5):
        super(Classifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1)
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.resnet(x)
        attention_map = self.attention(features)
        attention_weights = torch.softmax(attention_map.view(attention_map.size(0), -1), dim=1)
        attention_weights = attention_weights.view(attention_map.size())

        weighted_features = features * attention_weights
        pooled_features = torch.sum(weighted_features, dim=[2, 3])
        output = self.fc(pooled_features)
        
        return output, attention_map

BBPS_model = Classifier(num_classes=5)
BBPS_model.load_state_dict(torch.load(BBPSmodel_path, weights_only=True), strict=False)
BBPS_model = BBPS_model.to(device)
BBPS_model.eval()

categories = ['ileocecus', 'ascending_colon', 'transverse_colon', 'descending_colon', 'sigmoid_colon', 'rectum']

emission_map = pd.read_csv('./documents/emission_map.csv').set_index('Unnamed: 0')
transmit_map = pd.read_csv('./documents/transfer_map.csv').set_index('Unnamed: 0')

def infer(selected_video, st_time, root_dir):
    seq_df = pd.read_csv('./documents/seg_seq.csv')
    if selected_video in seq_df['video'].values:
        return 'The video has been processed!'
    now_frame = 0
    now_second = 0
    tot_frame = 0
    pre_seq = []
    BBPS_seq = []
    last_posb = [1, 0, 0, 0, 0, 0, 0, 0]
    last_pred = 0
    pred_book = []
    try:
        video = mp.VideoFileClip(f"{root_dir}/{selected_video.split('-')[0]}/{selected_video.split('-')[1]}.mp4")
    except:
        video = mp.VideoFileClip(f"{root_dir}/{selected_video.split('-')[0]}/{selected_video.split('-')[1]}.MP4")
        fps = round(video.fps)
    progress_bar = tqdm(video.iter_frames(), desc=f"{selected_video}", total=video.reader.nframes)
    for frame in progress_bar:
        now_frame += 1
        if now_frame % fps == 0:
            now_second += 1
        if now_second < st_time:
            continue

        tot_frame += 1
        image = frame[39:1044, 700:1863]

        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)

        BBPS_input_tensor = BBPS_transform(image)
        BBPS_input_tensor = BBPS_input_tensor.unsqueeze(0)

        with torch.no_grad():
            BBPS_input_tensor = BBPS_input_tensor.to(device)
            BBPSoutputs, _ = BBPS_model(BBPS_input_tensor)
            _, BBPSpredicted = torch.max(BBPSoutputs, 1)
        score = BBPSpredicted.tolist()[0] - 1

        with torch.no_grad():
            input_tensor= input_tensor.to(device)
            VJoutputs = VJmodel(input_tensor)
            _, VJpredicted = torch.max(VJoutputs, 1)

        prior_prob = [0, 0, 0, 0, 0, 0, 0, 0]
        posterior_prob = [0, 0, 0, 0, 0, 0, 0, 0]
        now_pred = 0
        if len(pred_book) >= 60:
            pred_book.pop(0)
        if VJpredicted.tolist()[0] == 0:
            now_pred = last_pred
            score = -1
        else:
            with torch.no_grad():
                input_tensor= input_tensor.to(device)
                outputs = classification_model(input_tensor)
                _, predicted = torch.max(outputs, 1)
            now_pred = predicted.tolist()[0]

        for i in range(6):
            for j in range(6):
                prior_prob[i] += transmit_map.loc[j, str(i)] * last_posb[j]
        for i in range(6):
            posterior_prob[i] = emission_map.loc[i, str(now_pred)] * prior_prob[i]
        posterior_prob = [x / sum(posterior_prob) for x in posterior_prob]
        last_posb = posterior_prob
        pred = posterior_prob.index(max(posterior_prob))
        last_pred = pred
        pred_book.append(pred)
        pre_seq.append(pred)
        BBPS_seq.append(score)
    
    seq_new_row = pd.DataFrame({'video': [selected_video], 'pred_seq': [pre_seq], 'BBPS': [BBPS_seq]})
    seq_df = pd.concat([seq_df, seq_new_row], ignore_index=True)
    seq_df.to_csv('./documents/seg_seq.csv', index=False)

    colors2 = plt.cm.tab10(pre_seq)

    # 创建一个图形和两个子图（2行1列）
    fig, axes = plt.subplots(1, 1, figsize=(10, 3), gridspec_kw={'height_ratios': [1]})
    axes.barh(0, len(pre_seq), left=np.arange(len(pre_seq)), color=colors2, height=0.5)
    axes.set_xlim(0, len(pre_seq))
    axes.get_yaxis().set_visible(False)
    axes.get_xaxis().set_visible(False)
    for spine in axes.spines.values():
        spine.set_visible(False)
    plt.savefig(f'./documents/figure_EndoVerse/{selected_video}.jpg', dpi=300, bbox_inches='tight') # selected_vedio
    return 'Done!'

if __name__ == "__main__":
    st_table = pd.read_csv('./documents/start_time.csv')
    for video in selected_video:
        st_time = st_table.loc[video]['st_time']
        infer(video, st_time)
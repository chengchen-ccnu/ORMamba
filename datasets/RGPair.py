import torch
import numpy as np
import os
import pickle
import random
import glob
# from os.path import join
from PIL import Image
from torchvideotransforms import video_transforms, volume_transforms
import pickle as pkl
from sklearn.preprocessing import RobustScaler
import cv2
import time
from models.i3d import I3D
from utils import misc


class RGPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform, clip_num=26, image_num=80, rand_st=True, action_type='Ball', score_type='Total_Score'):
        self.args = args
        self.data_path1 = args.data_root
        self.subset = subset
        self.transforms = transform
        # self.data_path2 = image_feat_path
        self.clip_num = clip_num
        self.image_num = image_num
        self.rand_st = rand_st
        self.frame_length = args.frame_length
        self.data_root = args.data_root
        self.voter_number = args.voter_number
        self.save_path = './data/Rhythmic_Gymnastics/i3d_features_RG.pkl'

        self.train_label_path = args.train_split
        self.train_label, self.train_split = self.read_label(self.train_label_path, score_type, action_type)


        self.test_label_path = args.test_split
        self.test_label, self.test_split = self.read_label(self.test_label_path, score_type, action_type)

        self.all_label = {**self.train_label, **self.test_label}

        self.robust_label = self.process_robust()

        self.train_video_name = self.read_name(args.train_split)
        self.test_video_name = self.read_name(args.test_split)

        self.train_transforms = video_transforms.Compose([
            video_transforms.RandomHorizontalFlip(),
            video_transforms.Resize((455, 256)),
            video_transforms.RandomCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.test_transforms = video_transforms.Compose([
            video_transforms.Resize((455, 256)),
            video_transforms.CenterCrop(224),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.backbone = I3D(num_classes=400, modality='rgb', dropout_prob=0.5).eval().cuda()
        self.backbone.load_state_dict(torch.load(args.pretrained_i3d_weight))

        self.video_feature = self.read_pickle(self.save_path)


        # self.frames_num = self.extract_frames()       //  将视频处理为帧图像
        # self.extract_features()           //  将帧提取特征保存


    def extract_frames(self):
        # 输入与输出路径
        video_root = './data/Rhythmic_Gymnastics/videos'
        frame_root = './data/Rhythmic_Gymnastics/frames'

        # 确保输出目录存在
        os.makedirs(frame_root, exist_ok=True)
        frames_num = {}
        frames_num['Ball'] = []
        frames_num['Clubs'] = []
        frames_num['Hoop'] = []
        frames_num['Ribbon'] = []
        # 获取所有视频文件
        videos = [v for v in os.listdir(video_root) if v.endswith('.mp4')]

        for video_name in videos:
            start_time = time.time()
            # === 解析视频名称 ===
            # 例如 Ball_001.mp4 -> project='Ball', idx='001'
            name, _ = os.path.splitext(video_name)
            try:
                project, idx = name.split('_')
            except ValueError:
                print(f"[WARN] 文件名格式不符合要求: {video_name}")
                continue

            # === 创建对应的输出路径 ===
            save_dir = os.path.join(frame_root, project, idx)
            os.makedirs(save_dir, exist_ok=True)

            # === 读取视频 ===
            video_path = os.path.join(video_root, video_name)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"[ERROR] 无法打开视频: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = 0

            # === 逐帧读取并保存 ===
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                # --- 等比例缩小 2 倍 ---
                h, w = frame.shape[:2]
                resized_frame = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                frame_name = f"frame_{frame_idx:04d}.jpg"
                save_path = os.path.join(save_dir, frame_name)
                cv2.imwrite(save_path, resized_frame)

            cap.release()
            frames_num[project].append(frame_idx)
            end_time = time.time()
            print(f"[INFO] {video_name} -> {frame_idx} frames saved to {save_dir}  time: {end_time - start_time} s")
        return frames_num

    @torch.no_grad()
    def extract_features(self):
        """
        从 ./data/Rhythmic_Gymnastics/frames/<project>/<id> 结构下提取每个视频的特征。
        保存格式为：{ 'Ball_001': feature_tensor, 'Ribbon_002': feature_tensor, ... }
        """
        frame_root = self.data_root
        feature_dict = {}
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        # 获取所有项目类别目录（Ball、Clubs、Hoop、Ribbon）
        projects = [p for p in os.listdir(frame_root) if os.path.isdir(os.path.join(frame_root, p))]

        cnt = 1
        for project in projects:
            project_dir = os.path.join(frame_root, project)
            # 每个项目下的编号文件夹（如 001, 002, ...）
            sample_ids = sorted([s for s in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, s))])

            for sample_id in sample_ids:
                video_key = f"{project}_{sample_id}"  # 例如 Ball_001
                start_time = time.time()
                print(f"第 {cnt} 个视频 [{video_key}] 开始处理")
                cnt += 1

                frames_path = os.path.join(project_dir, sample_id)
                image_list = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
                if len(image_list) == 0:
                    print(f"[WARN] {frames_path} contains no frames!")
                    continue

                # 判断train/test
                is_train = video_key in self.train_video_name
                transforms = self.train_transforms if is_train else self.test_transforms

                # --- 帧插值到固定长度 ---
                start_frame = int(os.path.basename(image_list[0])[6:10])  # 从 frame_0001.jpg 解析数字
                end_frame = int(os.path.basename(image_list[-1])[6:10])
                frame_list = np.linspace(start_frame, end_frame, self.frame_length).astype(int)

                # --- 加载所有帧 ---
                raw_imgs = []
                for idx in frame_list:
                    frame_path = os.path.join(frames_path, f"frame_{idx:04d}.jpg")
                    if not os.path.exists(frame_path):
                        continue
                    img = Image.open(frame_path).convert("RGB")
                    raw_imgs.append(img)
                if len(raw_imgs) == 0:
                    print(f"[WARN] {video_key} 没有有效帧，跳过。")
                    continue

                video_tensor = transforms(raw_imgs)  # 同步增强
                total_video = video_tensor.unsqueeze(0)  # [1,C,T,H,W]

                # --- clip 分段特征提取 ---
                start_idx = [i for i in range(0, 2300, 10)]  # 共230个clip
                num_clips = len(start_idx)
                batch_size = 230
                clip_features = []

                for j in range(0, num_clips, batch_size):
                    batch_clips = []
                    for i in start_idx[j:j + batch_size]:
                        clip = total_video[:, :, i:i + 16]
                        batch_clips.append(clip)
                    batch_clips = torch.cat(batch_clips, dim=0).cuda()

                    with torch.cuda.amp.autocast():
                        batch_feats = self.backbone(batch_clips)  # [batch_size,1024,1,1,1]
                    batch_feats = batch_feats.squeeze(-1).squeeze(-1).squeeze(-1)
                    batch_feats = batch_feats.unsqueeze(0)  # [1,batch_size,1024]
                    clip_features.append(batch_feats.cpu())

                    del batch_clips, batch_feats
                    torch.cuda.empty_cache()

                # --- 拼接特征 ---
                total_feature = torch.cat(clip_features, dim=1).squeeze(0)  # [1,num_clips,1024]
                feature_dict[video_key] = total_feature

                # --- 每个视频写入一次，防止中断丢失 ---
                with open(self.save_path, 'wb') as f:
                    pickle.dump(feature_dict, f)
                end_time = time.time()
                print(f"[INFO] {video_key} -> {total_feature.shape} 特征已保存，耗时：%.4f s" %(end_time - start_time))

        print(f"✅ 所有视频特征已保存至: {self.save_path}")

    def process_robust(self):
        robust_dict = {}
        train_name = self.train_split
        test_name = self.test_split
        all_name = train_name + test_name
        # 提取所有样本的分数（第三个元素）
        scores_1 = [self.all_label[item] for item in self.train_split]
        scores_2 = [self.all_label[item] for item in self.test_split]
        scores = scores_1 + scores_2
        scores_2d = [[s] for s in scores]  # RobustScaler 需要二维输入

        # 进行 RobustScaler 标准化
        scaler = RobustScaler()
        robust_scores = scaler.fit_transform(scores_2d)

        # 建立 index -> 标准化后分数 的映射
        for idx, robust_val in enumerate(robust_scores):
            robust_dict[all_name[idx]] = robust_val[0]

        # 保存并返回
        self.robust_label = robust_dict
        return self.robust_label

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def read_name(self, label_path):
        fr = open(label_path, 'r')
        names = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            names.append(line[0])
        return names

    def read_label(self, label_path, score_type, action_type):
        fr = open(label_path, 'r')
        idx = {'Difficulty_Score': 1, 'Execution_Score': 2, 'Total_Score': 3}
        labels = {}
        train_video_split = []
        for i, line in enumerate(fr):
            if i == 0:
                continue
            line = line.strip().split()
            labels[line[0]] = float(line[idx[score_type]])
            if action_type == line[0].split('_')[0]:
                train_video_split.append(line[0])
        return labels , train_video_split

    def delta(self):
        delta = []
        dataset = self.train_split.copy()
        for i in range(len(dataset)):
            for j in range(i+1,len(dataset)):
                delta.append(abs(self.all_label[dataset[i]] - self.all_label[dataset[j]]))
        return delta

    def __getitem__(self, idx):
        if self.subset == 'train':
            key = self.train_split[idx]
        elif self.subset == 'test':
            key = self.test_split[idx]
        data = {}
        if self.subset == 'test':
            # test phase
            # data['video'] = self.load_video(key , 'test')
            data['video'] = self.video_feature[key]
            data['final_score'] = self.all_label.get(key)

            train_file_list = self.train_split.copy()
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]

            # exemplar
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                # tmp['video'] = self.load_video(item , 'test')
                tmp['video'] = self.video_feature[item]
                tmp['final_score'] = self.all_label.get(item)
                target_list.append(tmp)

            return data, target_list
        else:
            # train phase
            data['video'] = self.video_feature[key]
            data['final_score'] = self.all_label.get(key)
            data['robust_score'] = self.robust_label[key]
            file_list = self.train_split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(key))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            # sample 2
            # target['video'] = self.load_video(sample_2, 'train')
            target['video'] = self.video_feature[sample_2]
            target['final_score'] = self.all_label.get(sample_2)
            target['robust_score'] = self.robust_label[sample_2]
            return data, target

    def __len__(self):
        if self.subset == 'train':
            return len(self.train_split)
        elif self.subset == 'test':
            return len(self.test_split)
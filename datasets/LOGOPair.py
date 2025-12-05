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

class LOGOPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)

        self.args = args
        self.subset = subset
        self.transforms = transform
        # some flags
        self.usingDD = args.usingDD
        self.dive_number_choosing = args.dive_number_choosing
        # file path
        self.label_path = args.label_path
        self.split_path = args.train_split
        self.feature_path = args.feature_path
        self.split = self.read_pickle(self.split_path)
        self.label_dict = self.read_pickle(self.label_path)
        self.data_root = args.data_root
        # setting
        self.temporal_shift = [args.temporal_shift_min, args.temporal_shift_max]
        self.voter_number = args.voter_number
        self.length = args.frame_length

        # build difficulty dict ( difficulty of each action, the cue to choose exemplar)
        self.difficulties_dict = {}
        self.dive_number_dict = {}
        if self.subset == 'test':
            self.split_path_test = args.test_split
            self.split_test = self.read_pickle(self.split_path_test)
            self.difficulties_dict_test = {}
            self.dive_number_dict_test = {}
        self.preprocess()
        self.robust_label = self.process_robust()

        self.choose_list = self.split.copy()
        if self.subset == 'test':
            self.dataset = self.split_test
        else:
            self.dataset = self.split
        self.video_feature = self.read_pickle(self.feature_path)

    def load_video(self, key , phase):
        length = self.length
        transforms = self.transforms
        frames_path = os.path.join(self.data_root, key[0], str(key[1]))
        image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
        start_frame = int(image_list[0].split("\\")[-1][:-4])
        end_frame = int(image_list[-1].split("\\")[-1][:-4])
        if phase == 'train':
            temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
            end_frame = end_frame + temporal_aug_shift
        frame_list = np.linspace(start_frame, end_frame, self.length).astype(int)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

        # return (image_list , image_frame_idx)
        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]
        return self.transforms(video)

    def load_idx(self, frames_path):
        length = self.length
        image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
        if len(image_list) >= length:
            start_frame = int(image_list[0].split("\\")[-1][:-4])
            end_frame = int(image_list[-1].split("\\")[-1][:-4])
            frame_list = np.linspace(start_frame, end_frame, length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(length)]
            return image_frame_idx
        else:
            T = len(image_list)
            img_idx_list = np.arange(T)
            img_idx_list = img_idx_list.repeat(2)
            idx_list = np.linspace(0, T * 2 - 1, length).astype(np.int)
            image_frame_idx = [img_idx_list[idx_list[i]] for i in range(length)]
            return image_frame_idx

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def load_boxes(self, key, image_frame_idx, out_size):  # T,N,4
        key_bbox_list = [(key[0], str(key[1]), str(i).zfill(4)) for i in image_frame_idx]
        N = self.num_boxes
        T = self.length
        H, W = out_size
        boxes = []
        for key_bbox in key_bbox_list:
            person_idx_list = []
            for i, item in enumerate(self.boxes_dict[key_bbox]['box_label']):
                if item == 'person':
                    person_idx_list.append(i)
            tmp_bbox = []
            tmp_x1, tmp_y1, tmp_x2, tmp_y2 = 0, 0, 0, 0
            for idx, person_idx in enumerate(person_idx_list):
                if idx < N:
                    box = self.boxes_dict[key_bbox]['boxes'][person_idx]
                    box[:2] -= box[2:] / 2
                    x, y, w, h = box.tolist()
                    x = x * W
                    y = y * H
                    w = w * W
                    h = h * H
                    tmp_x1, tmp_y1, tmp_x2, tmp_y2 = x, y, x + w, y + h
                    tmp_bbox.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))  # 1,4 x1,y1,x2,y2
            if len(person_idx_list) < N:
                step = len(person_idx_list)
                while step < N:
                    tmp_bbox.append(torch.tensor([tmp_x1, tmp_y1, tmp_x2, tmp_y2]).unsqueeze(0))  # 1,4
                    step += 1
            boxes.append(torch.cat(tmp_bbox).unsqueeze(0))  # 1,N,4
        boxes_tensor = torch.cat(boxes)
        return boxes_tensor

    def preprocess(self):
        for item in self.split:
            difficulty = self.label_dict.get(item)[0]
            if self.difficulties_dict.get(difficulty) is None:
                self.difficulties_dict[difficulty] = []
            self.difficulties_dict[difficulty].append(item)

        if self.subset == 'test':
            for item in self.split_test:
                difficulty = self.label_dict.get(item)[0]
                if self.difficulties_dict_test.get(difficulty) is None:
                    self.difficulties_dict_test[difficulty] = []
                self.difficulties_dict_test[difficulty].append(item)

    def process_robust(self):
        robust_dict = {}

        for difficulty, file_list in self.difficulties_dict.items():
            # 提取该难度下所有分数
            scores = [self.label_dict[fname][1] for fname in file_list]
            scores_2d = [[s] for s in scores]  # RobustScaler 需要二维

            scaler = RobustScaler()
            robust_scores = scaler.fit_transform(scores_2d)

            # 遍历样本
            for fname, robust_val in zip(file_list, robust_scores):
                robust_dict[fname] = robust_val[0]

        # 保存到类的属性中
        self.robust_label = robust_dict
        return self.robust_label

    def delta(self):
        delta = []
        dataset = self.split.copy()
        for i in range(len(dataset)):
            for j in range(i + 1, len(dataset)):
                delta.append(
                    abs(
                        self.label_dict[dataset[i]][1] -
                        self.label_dict[dataset[j]][1]))

        return delta

    def __getitem__(self, index):
        key = self.dataset[index]
        data = {}
        if self.subset == 'test':
            # test phase
            # data['video'] = self.load_video(key , 'test')
            data['video'] = self.video_feature[key]
            data['final_score'] = self.label_dict.get(key)[1]
            # DD---TYPE
            if self.label_dict.get(key)[0] == 'free':
                data['difficulty'] = 0
            elif self.label_dict.get(key)[0] == 'tech':
                data['difficulty'] = 1
            train_file_list = self.difficulties_dict[self.label_dict[key][0]]
            random.shuffle(train_file_list)
            choosen_sample_list = train_file_list[:self.voter_number]
            # goat
            # data = self.load_goat_data(data, key , 'test')

            # exemplar
            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                # tmp['video'] = self.load_video(item , 'test')
                tmp['video'] = self.video_feature[item]
                tmp['final_score'] = self.label_dict.get(item)[1]
                if self.label_dict.get(item)[0] == 'free':
                    tmp['difficulty'] = 0
                elif self.label_dict.get(item)[0] == 'tech':
                    tmp['difficulty'] = 1
                target_list.append(tmp)

            return data, target_list
        else:
            # train phase
            # data['video'] = self.load_video(key , 'train')
            data['video'] = self.video_feature[key]
            data['final_score'] = self.label_dict.get(key)[1]
            if self.label_dict.get(key)[0] == 'free':
                data['difficulty'] = 0
            elif self.label_dict.get(key)[0] == 'tech':
                data['difficulty'] = 1
            data['robust_score'] = self.robust_label[key]

            file_list = self.difficulties_dict[self.label_dict[key][0]].copy()  # @
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
            target['final_score'] = self.label_dict.get(sample_2)[1]
            if self.label_dict.get(sample_2)[0] == 'free':
                target['difficulty'] = 0
            elif self.label_dict.get(sample_2)[0] == 'tech':
                target['difficulty'] = 1
            target['robust_score'] = self.robust_label[sample_2]
            return data, target

    def __len__(self):
        return len(self.dataset)
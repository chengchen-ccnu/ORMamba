import torch
import numpy as np
import os
import pickle
import random
import glob
# from os.path import join
from PIL import Image

import pandas as pd
from sklearn.preprocessing import RobustScaler


class FineDivingPair_Dataset(torch.utils.data.Dataset):
    def __init__(self, args, subset, transform):
        random.seed(args.seed)
        self.subset = subset
        self.transforms = transform
        # using Difficult Degree
        self.usingDD = args.usingDD
        # some flags
        self.dive_number_choosing = args.dive_number_choosing
        # file path
        self.label_path = args.label_path
        self.split_path = args.train_split
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
        self.robust_label = {}
        if self.subset == 'test':
            self.split_path_test = args.test_split
            self.split_test = self.read_pickle(self.split_path_test)
            self.difficulties_dict_test = {}
            self.dive_number_dict_test = {}
        if self.usingDD:
            self.preprocess()
            self.check()
            self.robust_label = self.process_robust()

        self.choose_list = self.split.copy()
        if self.subset == 'test':
            self.dataset = self.split_test
            #   该样本在训练集找不到难度系数同样为1.5的参考样本，去掉
            del (self.dataset[249])
        else:
            self.dataset = self.split

    def load_video(self, video_file_name, phase):
        image_list = sorted(
            (glob.glob(os.path.join(self.data_root, video_file_name[0], str(video_file_name[1]), '*.jpg'))))

        start_frame = int(image_list[0].split("\\")[-1][:-4])
        end_frame = int(image_list[-1].split("\\")[-1][:-4])

        if phase == 'train':
            temporal_aug_shift = random.randint(self.temporal_shift[0], self.temporal_shift[1])
            end_frame = end_frame + temporal_aug_shift

        frame_list = np.linspace(start_frame, end_frame, self.length).astype(int)
        image_frame_idx = [frame_list[i] - start_frame for i in range(self.length)]

        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(self.length)]

        frames_labels = [self.label_dict.get(video_file_name)[4][i] for i in image_frame_idx]
        frames_catogeries = list(set(frames_labels))
        frames_catogeries.sort(key=frames_labels.index)
        transitions = [frames_labels.index(c) for c in frames_catogeries]


        _video = self.transforms(video)
        return _video , np.array([transitions[1]-1,transitions[-1]-1])

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

    def preprocess(self):
        if self.dive_number_choosing:
            # Dive Number
            for item in self.split:
                dive_number = self.label_dict.get(item)[0]
                if self.dive_number_dict.get(dive_number) is None:
                    self.dive_number_dict[dive_number] = []
                self.dive_number_dict[dive_number].append(item)

            if self.subset == 'test':
                for item in self.split_test:
                    dive_number = self.label_dict.get(item)[0]
                    if self.dive_number_dict_test.get(dive_number) is None:
                        self.dive_number_dict_test[dive_number] = []
                    self.dive_number_dict_test[dive_number].append(item)
        else:
            # DD
            for item in self.split:
                difficulty = self.label_dict.get(item)[2]
                if self.difficulties_dict.get(difficulty) is None:
                    self.difficulties_dict[difficulty] = []
                self.difficulties_dict[difficulty].append(item)

            if self.subset == 'test':
                for item in self.split_test:
                    difficulty = self.label_dict.get(item)[2]
                    if self.difficulties_dict_test.get(difficulty) is None:
                        self.difficulties_dict_test[difficulty] = []
                    self.difficulties_dict_test[difficulty].append(item)

    def check(self):
        if self.dive_number_choosing:
            # dive_number_dict
            for key in sorted(list(self.dive_number_dict.keys())):
                file_list = self.dive_number_dict[key]
                for item in file_list:
                    assert self.label_dict[item][0] == key

            if self.subset == 'test':
                for key in sorted(list(self.dive_number_dict_test.keys())):
                    file_list = self.dive_number_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item][0] == key
        else:
            # difficulties_dict
            for key in sorted(list(self.difficulties_dict.keys())):
                file_list = self.difficulties_dict[key]
                for item in file_list:
                    assert self.label_dict[item][2] == key

            if self.subset == 'test':
                for key in sorted(list(self.difficulties_dict_test.keys())):
                    file_list = self.difficulties_dict_test[key]
                    for item in file_list:
                        assert self.label_dict[item][2] == key

        print('check done')

    def delta(self):
        '''
            RT: builder group
        '''
        if self.usingDD:
            if self.dive_number_choosing:
                delta = []
                for key in list(self.dive_number_dict.keys()):
                    file_list = self.dive_number_dict[key]
                    for i in range(len(file_list)):
                        for j in range(i + 1, len(file_list)):
                            delta.append(abs(
                                self.label_dict[file_list[i]][1] / self.label_dict[file_list[i]][2] -
                                self.label_dict[file_list[j]][1] / self.label_dict[file_list[j]][2]))
            else:
                delta = []
                for key in list(self.difficulties_dict.keys()):
                    file_list = self.difficulties_dict[key]
                    for i in range(len(file_list)):
                        for j in range(i + 1, len(file_list)):
                            delta.append(abs(
                                self.label_dict[file_list[i]][1] / self.label_dict[file_list[i]][2] -
                                self.label_dict[file_list[j]][1] / self.label_dict[file_list[j]][2]))
        else:
            delta = []
            dataset = self.split.copy()
            for i in range(len(dataset)):
                for j in range(i + 1, len(dataset)):
                    delta.append(
                        abs(
                            self.label_dict[dataset[i]][1] -
                            self.label_dict[dataset[j]][1]))

        return delta

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

    def __getitem__(self, index):
        sample_1 = self.dataset[index]
        data = {}
        if self.subset == 'test':
            # test phase
            data['video'] , data['transits'] = self.load_video(sample_1, 'test')
            data['final_score'] = self.label_dict.get(sample_1)[1]
            data['difficulty'] = self.label_dict.get(sample_1)[2]
            data['completeness'] = (data['final_score'] / data['difficulty'])
            # data['robust_score'] = self.robust_label[sample_1]

            if self.usingDD:
                # NOTE: using Dive Number to choose
                if self.dive_number_choosing:
                    train_file_list = self.dive_number_dict[self.label_dict[sample_1][0]]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[:self.voter_number]
                else:
                    # choose a list of sample in training_set
                    train_file_list = self.difficulties_dict[self.label_dict[sample_1][2]]
                    random.shuffle(train_file_list)
                    choosen_sample_list = train_file_list[:self.voter_number]
            else:
                train_file_list = self.choose_list
                random.shuffle(train_file_list)
                choosen_sample_list = train_file_list[:self.voter_number]

            target_list = []
            for item in choosen_sample_list:
                tmp = {}
                tmp['video'] , tmp['transits'] = self.load_video(item, 'test')
                tmp['final_score'] = self.label_dict.get(item)[1]
                tmp['difficulty'] = self.label_dict.get(item)[2]
                tmp['completeness'] = (tmp['final_score'] / tmp['difficulty'])
                # tmp['robust_score'] = self.robust_label[item]
                # print(tmp)
                target_list.append(tmp)

            return data, target_list
        else:
            # train phase
            data['video'] , data['transits'] = self.load_video(sample_1, 'train')
            data['final_score'] = self.label_dict.get(sample_1)[1]
            data['difficulty'] = self.label_dict.get(sample_1)[2]
            data['completeness'] = (data['final_score'] / data['difficulty'])
            data['robust_score'] = self.robust_label[sample_1]

            # choose a sample
            if self.usingDD:
                # did not using a pytorch sampler, using diff_dict to pick a video sample
                if self.dive_number_choosing:
                    # NOTE: using Dive Number to choose
                    file_list = self.dive_number_dict[self.label_dict[sample_1][0]].copy()
                else:
                    # all sample owning same difficulties
                    file_list = self.difficulties_dict[self.label_dict[sample_1][2]].copy()
            else:
                # randomly
                file_list = self.split.copy()
            # exclude self
            if len(file_list) > 1:
                file_list.pop(file_list.index(sample_1))
            # choosing one out
            idx = random.randint(0, len(file_list) - 1)
            sample_2 = file_list[idx]
            target = {}
            # sample 2
            target['video'] , target['transits'] = self.load_video(sample_2, 'train')
            target['final_score'] = self.label_dict.get(sample_2)[1]
            target['difficulty'] = self.label_dict.get(sample_2)[2]
            target['completeness'] = (target['final_score'] / target['difficulty'])
            target['robust_score'] = self.robust_label[sample_2]
            return data, target

    def __len__(self):
        return len(self.dataset)


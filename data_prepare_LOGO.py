import torch
import numpy as np
import os
import pickle
import random
import glob
from PIL import Image
from torchvideotransforms import video_transforms, volume_transforms
from sklearn.preprocessing import RobustScaler
from models.i3d import I3D
from utils import parser

class Extract_feature(torch.utils.data.Dataset):
    def __init__(self, args, subset , save_path = './data/LOGO/i3d_features_LOGO.pkl'):
        random.seed(args.seed)

        self.args = args
        self.subset = subset
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

        self.backbone = I3D(num_classes=400, modality='rgb', dropout_prob=0.5).eval().cuda()
        self.backbone.load_state_dict(torch.load(args.pretrained_i3d_weight))
        self.save_path = save_path
        self.length = 5406
        self.clip_len = 16
        self.clip_stride = 10
        self.train_transforms = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        self.test_transforms = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        self.extract_features()




    @torch.no_grad()
    def extract_features(self):
        feature_dict = {}
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        cnt = 1
        for key in self.label_dict.keys():
            print("No."+str(cnt)+'Begin Process')
            video_name, segment_idx = key
            frames_path = os.path.join(self.data_root, video_name, str(segment_idx))
            image_list = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
            if len(image_list) == 0:
                print(f"[WARN] {frames_path} contains no frames!")
                continue

            # train/test
            is_train = key in self.dataset
            transforms = self.train_transforms if is_train else self.test_transforms


            start_frame = int(os.path.basename(image_list[0])[:-4])
            end_frame = int(os.path.basename(image_list[-1])[:-4])
            frame_list = np.linspace(start_frame, end_frame, self.length).astype(int)


            raw_imgs = []
            for idx in frame_list:
                frame_path = os.path.join(frames_path, f"{idx:04d}.jpg")
                if not os.path.exists(frame_path):
                    continue
                img = Image.open(frame_path)  # [C,H,W] uint8
                raw_imgs.append(img)
            if len(raw_imgs) == 0:
                continue
            video_tensor = transforms(raw_imgs)
            total_video = video_tensor.unsqueeze(0)  # [1,C,T,H,W]


            start_idx = [i for i in range(0, 5400, self.clip_stride)]
            num_clips = len(start_idx)
            batch_size = 270
            clip_features = []

            for j in range(0, num_clips, batch_size):

                batch_clips = []
                for i in start_idx[j:j + batch_size]:
                    clip = total_video[:, :, i:i + self.clip_len]  # [1,3,16,224,224]
                    batch_clips.append(clip)
                batch_clips = torch.cat(batch_clips, dim=0).cuda()  # [batch_size,3,16,224,224]

                # forward
                with torch.cuda.amp.autocast():
                    batch_feats = self.backbone(batch_clips)  # [batch_size,1024,1,1,1]
                batch_feats = batch_feats.squeeze(-1).squeeze(-1).squeeze(-1)  # [batch_size,1024]
                batch_feats = batch_feats.unsqueeze(0)  # [1,batch_size,1024]
                clip_features.append(batch_feats.cpu())

                del batch_clips, batch_feats
                torch.cuda.empty_cache()


            total_feature = torch.cat(clip_features, dim=1)
            feature_dict[key] = total_feature

            with open(self.save_path, 'wb') as f:
                pickle.dump(feature_dict, f)
            print("No." + str(cnt) + 'done')
            cnt += 1

        print(f"all features saved: {self.save_path}")

    def read_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        return pickle_data

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

            scores = [self.label_dict[fname][1] for fname in file_list]
            scores_2d = [[s] for s in scores]

            scaler = RobustScaler()
            robust_scores = scaler.fit_transform(scores_2d)


            for fname, robust_val in zip(file_list, robust_scores):
                robust_dict[fname] = robust_val[0]

        # 保存到类的属性中
        self.robust_label = robust_dict
        return self.robust_label

if __name__ == '__main__':
    args = parser.get_args()
    parser.setup(args)
    e = Extract_feature(args , 'train' , save_path= './data/LOGO/i3d_features_LOGO.pkl')
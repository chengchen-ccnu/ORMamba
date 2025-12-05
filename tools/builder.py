import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import PIL

# optimizer
import torch.optim as optim
import random
import traceback
# model
from models.Backbone import I3D_backbone

# utils
from utils.misc import import_class
from utils.Group_helper import Group_helper
from torchvideotransforms import video_transforms, volume_transforms
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.nn.functional as F



def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def dataset_builder(args):
    try:
        train_trans, test_trans = get_video_trans()
        Dataset = import_class("datasets." + args.benchmark)
        train_dataset = Dataset(args, transform=train_trans, subset='train')
        test_dataset = Dataset(args, transform=test_trans, subset='test')
        return train_dataset, test_dataset
    except Exception as e:
        traceback.print_exc()
        exit()

import torch.nn.init as init
import torch.nn as nn
class Regressor(nn.Module):
    def __init__(self, input_dim=1153, output_dim=1):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # 第一层，输入维度1024，输出512
        self.fc2 = nn.Linear(512, 256)        # 第二层，输入512，输出256
        self.fc3 = nn.Linear(256, output_dim) # 输出层
        self.relu = nn.ReLU()                 # 激活函数
        self.softmax = nn.Softmax(dim=1)      # Softmax 激活函数，用于输出概率分布
        # 权重初始化
        self._initialize_weights()
    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层 + 激活函数
        x = self.relu(self.fc2(x))  # 第二层 + 激活函数
        x = torch.sigmoid(self.fc3(x))            # 输出层
        # x = self.softmax(x)         # softmax 用于分类输出
        return x

    def _initialize_weights(self):
        # He 初始化：适用于 ReLU 激活函数
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')  # 使用 He 初始化第一层
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')  # 使用 He 初始化第二层
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')  # 使用 He 初始化输出层

class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(1153, 512)  # 第一层，输入维度1024，输出512
        self.fc2 = nn.Linear(512, 256)  # 第二层，输入512，输出256
        self.fc3 = nn.Linear(256, num_classes-1)  # 输出层
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层 + 激活函数
        x = self.relu(self.fc2(x))  # 第二层 + 激活函数
        logits = self.fc3(x)  # 输出层
        return logits

class Classifier_mistake(torch.nn.Module):
    def __init__(self, in_dim=576, hidden_dim=64, dropout=0.1):
        super(Classifier_mistake, self).__init__()
        # 简单的两层 MLP
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # 输出 1 个值 (logit)

    def forward(self, x):
        """
        x: [batch_size, in_dim]
        return: [batch_size, ] 二分类概率
        """
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        logit = self.fc2(x).squeeze(-1)       # [batch_size]
        prob = torch.sigmoid(logit)           # [0,1] 概率
        return prob




def model_builder(args):
    base_model = I3D_backbone(I3D_class = 400 , args = args)
    base_model.load_pretrain(args.pretrained_i3d_weight)
    # Regressor = RegressTree(
    #                     in_channel = 2 * base_model.get_feature_dim() + 1,
    #                     hidden_channel = 256,
    #                     depth = args.RT_depth)
    regressor = Regressor()
    classifier = Classifier(num_classes=16)
    classifier_mistake = Classifier_mistake()
    return base_model, regressor,classifier , classifier_mistake

def build_group(dataset_train, args):
    delta_list = dataset_train.delta()
    group = Group_helper(delta_list, args.RT_depth, Symmetrical = True, Max = args.score_range, Min = 0)
    return group

def build_opti_sche(base_model, regressor,classifier,classifier_mistake, n_iter_per_epoch,  args):
    if args.optimizer == 'Adam':
        optimizer = optim.Adam([
            {'params': base_model.parameters(), 'lr': args.base_lr * args.lr_factor * 0.1},
            # {'params': base_model.backbone.parameters(), 'lr': args.base_lr * args.lr_factor},
            # {'params': base_model.mamba.parameters(), 'lr': args.base_lr * args.lr_factor},
            {'params': regressor.parameters()},
            {'params': classifier.parameters()},
            {'params': classifier_mistake.parameters()}
        ], lr = args.base_lr , weight_decay = args.weight_decay)
    else:
        raise NotImplementedError()

    if args.use_scheduler:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.max_epoch * n_iter_per_epoch,
            lr_min=args.base_lr * args.lr_factor,
            warmup_lr_init=args.base_lr * args.lr_factor,
            warmup_t=args.warmup_epochs * n_iter_per_epoch,
            cycle_limit=15,
            t_in_epochs=False,
        )
    else:
        scheduler = None
    return optimizer, scheduler


def resume_train(base_model, regressor,classifier,classifier_mistake,optimizer, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0, 0, 1000, 1000
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path,map_location='cpu')
    # parameter resume of base model
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)

    classifier_ckpt = {k.replace("module.", ""): v for k, v in state_dict['classifier'].items()}
    classifier.load_state_dict(classifier_ckpt)


    classifier_mistake_ckpt = {k.replace("module.", ""): v for k, v in state_dict['classifier_mistake'].items()}
    classifier_mistake.load_state_dict(classifier_mistake_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])


    # parameter
    start_epoch = state_dict['epoch'] + 1
    epoch_best = state_dict['epoch_best']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best, rho_best, L2_min, RL2_min



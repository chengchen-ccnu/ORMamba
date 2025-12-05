import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch
import time
import numpy as np
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

def network_forward_train(base_model, regressor,classifier, classifier_mistake , pred_scores, video_1, label_1, video_2, label_2, robust_score_1 , robust_score_2,  diff, group, mse, auc_loss , optimizer, scheduler, opti_flag, epoch, batch_idx, batch_num, args , epoch_loss , n_iter_per_epoch):
    loss = 0.0
    loss_auc = 0.0
    start = time.time()
    combined_feature_1, combined_feature_2 , feature_1 , feature_2 = base_model(video_1, video_2, label=[label_1, label_2], is_train=True,theta=args.score_range)

    combined_feature = torch.cat((combined_feature_1, combined_feature_2), 0)
    ############# mistake prediction #############
    if robust_score_1 is not None:
        robust_score = torch.cat((robust_score_1, robust_score_2), 0)
        robust_score_label = torch.where(robust_score >= args.mistaken_threshold, 0, 1)
        if (robust_score_label.sum(0) > 0 and robust_score_label.sum(0) < robust_score_label.shape[0]):
            mistake_pred_1 = classifier_mistake(feature_1)
            mistake_pred_2 = classifier_mistake(feature_2)
            mistake_pred = torch.cat((mistake_pred_1, mistake_pred_2), 0).unsqueeze(1)
            loss_auc = auc_loss(mistake_pred, robust_score_label)
        loss += loss_auc

    ############# CORN Ordinal Regression #############
    logits = classifier(combined_feature)
    class_labels_pred = corn_label_from_logits(logits)

    delta_scores = torch.cat((label_2 - label_1, label_1 - label_2),0)
    group_tensor = torch.tensor(group.Group, dtype=torch.float32).cuda()  # (16,2)
    lows = group_tensor[:, 0]  # (16,)
    highs = group_tensor[:, 1]  # (16,)
    scores = delta_scores.view(-1, 1)  # (B,1)
    mask = (scores >= lows) & (scores < highs)
    mask[:, -1] |= (scores[:, 0] == highs[-1])
    class_labels = mask.float().argmax(dim=1)  # (B,)

    loss += corn_loss(logits, class_labels, len(group.Group))


    delta_pred = regressor(combined_feature)
    intervals = group_tensor[class_labels]  # (B,2)
    lows = intervals[:, 0].unsqueeze(1)  # (B,1)
    highs = intervals[:, 1].unsqueeze(1)  # (B,1)
    r = (delta_scores - lows) / (highs - lows)  # (B,1)
    r = torch.clamp(r, 0.0, 1.0)


    loss += mse(delta_pred , r)

    intervals = group_tensor[class_labels_pred]  # (B,2)
    lows = intervals[:, 0].unsqueeze(1)  # (B,1)
    highs = intervals[:, 1].unsqueeze(1)  # (B,1)


    relative_scores = lows + (highs - lows) * delta_pred  # (B,1)

    loss.backward()

    epoch_loss.append(loss.item())

    if opti_flag:
        optimizer.step()
        optimizer.zero_grad()

    if scheduler:
        scheduler.step_update(epoch * n_iter_per_epoch + batch_idx)

    end = time.time()
    batch_time = end - start
    if (batch_idx) % args.print_freq == 0:
        start_idx = batch_idx - args.print_freq
        avg_loss = sum(epoch_loss[start_idx:batch_idx]) / args.print_freq
        print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f \t lr3 : %0.5f' % (
        epoch, args.max_epoch, batch_idx, batch_num, batch_time, avg_loss, optimizer.param_groups[0]['lr'],
        optimizer.param_groups[1]['lr'] , optimizer.param_groups[2]['lr']))

    if (batch_idx == batch_num) and (batch_idx % args.print_freq != 0):
        remaining = batch_idx % args.print_freq
        start_idx = batch_idx - remaining
        avg_loss = sum(epoch_loss[start_idx:batch_idx]) / remaining
        print('[Training][%d/%d][%d/%d] \t Batch_time %.2f \t Batch_loss: %.4f \t lr1 : %0.5f \t lr2 : %0.5f \t lr3 : %0.5f' % (
        epoch, args.max_epoch, batch_idx, batch_num, batch_time, avg_loss, optimizer.param_groups[0]['lr'],
        optimizer.param_groups[1]['lr'] , optimizer.param_groups[2]['lr']))

    # evaluate result of training phase
    relative_scores = relative_scores[relative_scores.shape[0] // 2:]
    if args.benchmark == 'LOGO':
        score = relative_scores.cuda() + label_2
    elif args.benchmark == 'FineDiving':
        if args.usingDD:
            score = (relative_scores.cuda() + label_2)  * diff
        else:
            score = relative_scores.cuda() + label_2
    else:
        raise NotImplementedError()
    pred_scores.extend([i.item() for i in score])

def network_forward_test(base_model, regressor,classifier, classifier_mistake , pred_scores, video_1, video_2_list, label_1 , label_2_list, diff, group, args):
    score = 0

    for video_2, label_2 in zip(video_2_list, label_2_list):
        combined_feature = base_model(video_1, video_2, label=[label_2], is_train=False, theta=args.score_range)
        logits = classifier(combined_feature)
        class_labels_pred = corn_label_from_logits(logits)


        delta_pred = regressor(combined_feature)


        group_tensor = torch.tensor(group.Group, dtype=torch.float32).cuda()
        intervals = group_tensor[class_labels_pred]  # (B,2)
        lows = intervals[:, 0].unsqueeze(1)  # (B,1)
        highs = intervals[:, 1].unsqueeze(1)  # (B,1)


        relative_scores = lows + (highs - lows) * delta_pred  # (B,1)
        relative_scores = relative_scores[relative_scores.shape[0] // 2:]
        if args.benchmark == 'LOGO':
            score += relative_scores.cuda() + label_2
        elif args.benchmark == 'FineDiving':
            if args.usingDD:
                score.append((relative_scores.cuda() + label_2)  * diff)
            else:
                score += relative_scores.cuda() + label_2
        else:
            raise NotImplementedError()
    pred_scores.extend([i.item() / len(video_2_list) for i in score])


def save_checkpoint(base_model, regressor,classifier,classifier_mistake, optimizer, epoch, epoch_best, rho_best, L2_min, RL2_min, exp_name, args):
    torch.save({
                'base_model' : base_model.state_dict(),
                'regressor' : regressor.state_dict(),
                'classifier': classifier.state_dict(),
                'classifier_mistake': classifier_mistake.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'epoch_best': epoch_best,
                'rho_best' : rho_best,
                'L2_min' : L2_min,
                'RL2_min' : RL2_min,
                }, os.path.join(args.experiment_path, exp_name + '.pth'))


def save_outputs(pred_scores, true_scores, prefix, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred_' + prefix + '.npy')
    save_path_true = os.path.join(args.experiment_path, 'true_' + prefix + '.npy')
    np.save(save_path_pred ,pred_scores)
    np.save(save_path_true ,true_scores)

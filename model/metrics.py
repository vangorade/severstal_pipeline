# system

# libraries
import numpy as np
import torch
# modules


def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds


def metric(probability,
           truth,
           threshold=0.5,
           reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''

    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        # dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        # dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        # dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


def F_score(logit,
            label,
            threshold=0.5,
            beta=2):
    '''Calculates metrice F score '''

    # prob = torch.sigmoid(logit)
    batch_size = len(label)
    with torch.no_grad():
        logit = logit.view(batch_size, -1)
        label = label.view(batch_size, -1)
        assert(logit.shape == label.shape)

        prob = logit > threshold
        label = label > threshold

        TP = (prob * label).sum(1).float()
        # TN = ((np.logical_not(prob)) * (np.logical_not(label))).sum(1).float()
        FP = (prob * (np.logical_not(label))).sum(1).float()
        FN = ((np.logical_not(prob)) * label).sum(1).float()

        precision = torch.mean(TP / (TP + FP + 1e-12))
        recall = torch.mean(TP / (TP + FN + 1e-12))
        F2 = (1 + beta**2) * precision * recall / \
            (beta**2 * precision + recall + 1e-12).int()
    return F2.mean(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_constrastive_loss(data_dict, config, no_reference=False):
    """ Compute constrastive loss

    Args:
        data_dict: dict (read-only)

    Returns:
        constrastive loss
    """
    positive_labels = data_dict['positive_labels']  # (max_lang_num)
    sim_l2p = data_dict['sim_l2p']  # (max_lang_num, num_proposal)
    sim_p2l = data_dict['sim_p2l']  # (num_proposal, max_lang_num)

    contrast_mask = data_dict['contrast_mask']  # (batch_size, max_lang_num)
    batch_size, max_lang_num = contrast_mask.shape
    contrast_mask = contrast_mask.view(batch_size*max_lang_num)

    sim_l2p = sim_l2p[:, positive_labels]  # (max_lang_num, max_lang_num)
    sim_p2l = sim_p2l[positive_labels, :]  # (max_lang_num, max_lang_num)

    sim_p2l = sim_p2l[contrast_mask][:, contrast_mask]
    sim_l2p = sim_l2p[contrast_mask][:, contrast_mask]

    # sim_l2p = sim_l2p.masked_fill(contrast_mask, -1e9) # masked_fill
    # sim_p2l = sim_p2l.masked_fill(contrast_mask, -1e9) # masked_fill

    target = torch.eye(sim_l2p.shape[0]).long().cuda()
    loss_l2p = -torch.sum(F.log_softmax(sim_l2p, dim=1)*target, dim=1).mean()
    loss_p2l = -torch.sum(F.log_softmax(sim_p2l, dim=1)*target, dim=1).mean()

    loss = loss_l2p + loss_p2l
    loss = loss / batch_size
    return loss
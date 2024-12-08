import torch.nn.functional as F
def compute_answer_classification_loss(data_dict):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """
    if "answer_cat_scores" in data_dict:
        #  data_dict["answer_cat_scores"]: batch_size, num_answers
        loss_answer = F.binary_cross_entropy_with_logits(data_dict["answer_scores"], data_dict["answer_cat_scores"], reduction='sum') / data_dict["answer_scores"].shape[0]
    else:
        loss_answer = F.cross_entropy(data_dict["answer_scores"], data_dict["answer_cat"])
    return loss_answer
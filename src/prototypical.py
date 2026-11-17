import torch
import torch.nn.functional as F


def compute_prototypes(support_embeddings, support_labels, n_way):
    """Prototype of each class = mean embedding of its support samples"""
    prototypes = torch.zeros(n_way, support_embeddings.size(1), device=support_embeddings.device)
    for c in range(n_way):
        class_mask = (support_labels == c)
        prototypes[c] = support_embeddings[class_mask].mean(dim=0)
    return prototypes


def prototypical_loss(support_emb, support_labels, query_emb, query_labels, n_way):
    """
    Loss based on the Euclidean distance from each query to each class prototype.
    Smallest distance = highest probability (negative distance used as the logit).
    """
    prototypes = compute_prototypes(support_emb, support_labels, n_way)

    # Euclidean distance between each query and each prototype: (Q, n_way)
    dists = torch.cdist(query_emb, prototypes, p=2)
    logits = -dists  # smaller distance = more similar = larger logit

    loss = F.cross_entropy(logits, query_labels)
    preds = logits.argmax(dim=1)
    acc = (preds == query_labels).float().mean().item()

    return loss, acc

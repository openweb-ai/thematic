"""
Standalone functions extracted from "sentence-transformers.util" so that we can use
the fast clustering function from "sentence-transformers" w/o installing the entire
package.

Original Source:
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.pyhttps://github.com/UKPLab/sentence-transformers/blob/3575fe77916d5e10d2349b428adc8790e17e98c3/sentence_transformers/util.py

Modified by:
eterna2 <eterna2@hotmail.com>
marc-chan <marc.w.chan@gmail.com>
"""
from typing import List, Union

import numpy as np
import torch


def cos_sim(tensor_a: torch.Tensor, tensor_b: torch.Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(tensor_a, torch.Tensor):
        tensor_a = torch.tensor(tensor_a)

    if not isinstance(tensor_b, torch.Tensor):
        tensor_b = torch.tensor(tensor_b)

    if (len(tensor_a) == 0) or (len(tensor_b) == 0):
        return torch.tensor([])

    if len(tensor_a.shape) == 1:
        tensor_a = tensor_a.unsqueeze(0)

    if len(tensor_b.shape) == 1:
        tensor_b = tensor_b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(tensor_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(tensor_b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def community_detection_from_conjugate(
    conjugate: Union[np.ndarray, torch.Tensor],
    member_counts: Union[List, np.ndarray, torch.Tensor] = None,
    threshold: float = 0.75,
    min_community_size: int = 10,
    init_max_size: int = 1000,
):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.

    Adapted from "sentence-transformers.util.community_detection", with modification to allow member counts
    to be passed in for multilevel clustering. Separate calculation of conjugate to allow custom similarity
    matrices to be used.
    """

    if not isinstance(conjugate, torch.Tensor):
        conjugate = torch.Tensor(conjugate)

    if member_counts is None:
        member_counts = np.ones(len(conjugate), dtype=int)
    else:
        member_counts = np.array(member_counts)

    if len(member_counts) != conjugate.shape[0]:
        raise ValueError("`member_counts` and `conjugate` should be of equal length.")

    # Maximum size for community
    init_max_size = min(init_max_size, conjugate.shape[0])

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(conjugate.shape[0]):
        abv_thres_idx = (conjugate[i] >= threshold).argwhere().flatten()
        size = member_counts[abv_thres_idx].sum()

        if size >= min_community_size:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = conjugate[i].topk(
                k=init_max_size, largest=True
            )
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(conjugate[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=len, reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities


def community_detection(
    embeddings: Union[np.ndarray, torch.Tensor],
    member_counts: Union[List, np.ndarray, torch.Tensor] = None,
    threshold: float = 0.75,
    min_community_size: int = 10,
    init_max_size: int = 1000,
):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.

    Adapted from "sentence-transformers.util.community_detection", with modification to allow member counts
    to be passed in for multilevel clustering. Separate calculation of conjugate to allow custom similarity
    matrices to be used.
    """

    conjugate = cos_sim(embeddings, embeddings)
    unique_communities = community_detection_from_conjugate(
        conjugate,
        member_counts=member_counts,
        threshold=threshold,
        min_community_size=min_community_size,
        init_max_size=init_max_size,
    )
    return unique_communities

"""
Standalone functions extracted from "sentence-transformers.util" so that we can use
the fast clustering function from "sentence-transformers" w/o installing the entire
package.

Original Source:
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.pyhttps://github.com/UKPLab/sentence-transformers/blob/3575fe77916d5e10d2349b428adc8790e17e98c3/sentence_transformers/util.py

Modified by:
eterna2 <eterna2@hotmail.com>
"""
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

    if len(tensor_a.shape) == 1:
        tensor_a = tensor_a.unsqueeze(0)

    if len(tensor_b.shape) == 1:
        tensor_b = tensor_b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(tensor_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(tensor_b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def community_detection(
    embeddings, threshold=0.75, min_community_size=10, init_max_size=1000
):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

    # Maximum size for community
    init_max_size = min(init_max_size, len(embeddings))

    # Compute cosine similarity scores
    cos_scores = cos_sim(embeddings, embeddings)

    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(
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
                for idx, val in enumerate(cos_scores[i].tolist()):
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

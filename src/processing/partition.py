from numpy import inf
import numpy as np

def get_adjacency_by_group(skeleton_model, partition_groups):
    num_node = skeleton_model['num_nodes']
    kernel_size = 6
    adj_matrix = np.zeros((kernel_size, num_node, num_node))

    for node_idx in range(num_node):
        kernel_idx = partition_groups[node_idx]['limb']
        for neigh_idx in skeleton_model[node_idx]:
            adj_matrix[kernel_idx, node_idx, neigh_idx] = 1

    norm_coeficient = np.einsum('knm->km', adj_matrix)
    norm_coeficient = 1/norm_coeficient
    norm_coeficient[norm_coeficient == inf] = 0

    temp = np.einsum('knm->kmn', adj_matrix)
    for k in range(kernel_size):
        for n in range(num_node):
            temp[k, n, :] = temp[k, n, :] * norm_coeficient[k, n]
    adj_matrix = np.einsum('kmn->knm', temp)
    return adj_matrix, adj_matrix.shape[0]


def get_adjacency_by_distance(skeleton_model, partition_groups):
    num_node = skeleton_model['num_nodes']
    adj_matrix = np.zeros((3, num_node, num_node))

    for node_idx in range(num_node):
        s_center_distance = partition_groups[node_idx]['distance']
        for neigh_idx in skeleton_model[node_idx]:
            t_center_distance = partition_groups[neigh_idx]['distance']
            if s_center_distance < t_center_distance:
                adj_matrix[0, node_idx, neigh_idx] = 1
            elif s_center_distance > t_center_distance:
                adj_matrix[1, node_idx, neigh_idx] = 1
            elif s_center_distance == t_center_distance:
                adj_matrix[2, node_idx, neigh_idx] = 1

    norm_coeficient = np.einsum('knm->km', adj_matrix)
    norm_coeficient = 1/norm_coeficient
    norm_coeficient[norm_coeficient == inf] = 0

    temp = np.einsum('knm->kmn', adj_matrix)
    for k in range(3):
        for n in range(num_node):
            temp[k, n, :] = temp[k, n, :] * norm_coeficient[k, n]
    adj_matrix = np.einsum('kmn->knm', temp)
    return adj_matrix, 3
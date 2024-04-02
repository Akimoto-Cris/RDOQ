import numpy as np

class huffman_coding:

    def cal_huffman_code_length(self, tensor_data):
        data = tensor_data.cpu().numpy().reshape(-1)
        values, counts = np.unique(data, return_counts=True)
        counts = counts / counts.sum()
        counts.sort()
        n_alphabeta = len(counts)
        code_length = 0.0

        for i in range(0 , n_alphabeta - 1):
            node_1 = counts[0]
            node_2 = counts[1]
            node_combined = node_1 + node_2
            code_length += node_combined
            counts[1] = node_combined
            counts = counts[1:]
            counts.sort()

        return code_length

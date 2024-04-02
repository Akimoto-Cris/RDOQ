import numpy as np
import random

def generate_codebook(node, seq, id):
    codebook = []
    if node[2] == None:
        codebook.append(seq)
        node[3] = id
        return codebook, id + 1
    else:
        nxt_id = id
        for i in range(0, len(node[2])):
            _seq = list(seq)
            _seq.append(i)
            codewords, nxt_id = generate_codebook(node[2][i], _seq, nxt_id)
            for j in range(0, len(codewords)):
                codebook.append(codewords[j])
        #print(codebook)
        return codebook, nxt_id

def create_compute_cal_size(t, n):
    total_len = 0
    total_ele = 0

    for i in range(len(t)):
        [codeword, prob] = t[i]
        total_len += prob * n
        total_ele += prob * len(codeword)

    return total_len / total_ele

def ave_tunstall_code_size(alphabet, dist, n):
    size = len(alphabet)
    iterations = int((2 ** n - size) / (size - 1))
    t = []
    tunstall_tree = []

    for i, s in enumerate(alphabet):
        codeword = []
        codeword.append(s)
        t.append([codeword, dist[i]])
        tunstall_tree.append([s, dist[i], None, None])

    for _ in range(iterations):
        d = max(t, key=lambda p: p[1])
        ind = t.index(d)
        seq, seqProb = d
        for i, s in enumerate(alphabet):
            codeword = list(seq)
            codeword.append(s)
            t.append([codeword, seqProb * dist[i]])
        for i in range(0, len(seq)):
            if i == 0:
                cur = tunstall_tree[seq[i]]
            else:
                cur = cur[2][seq[i]]
        son_node = []
        for i, s in enumerate(alphabet):
            son_node.append([s, seqProb * dist[i], None, None])
        cur[2] = son_node
        del t[ind]

    return create_compute_cal_size(t, n)


def create_tunstall_tree(alphabet, dist, n):
    size = len(alphabet)
    iterations = int((2 ** n - size) / (size - 1))
    t = []
    tunstall_tree = []

    for i, s in enumerate(alphabet):
        codeword = []
        codeword.append(s)
        t.append([codeword, dist[i]])
        tunstall_tree.append([s, dist[i], None, None])

    for _ in range(iterations):
        d = max(t, key=lambda p: p[1])
        ind = t.index(d)
        seq, seqProb = d
        for i, s in enumerate(alphabet):
            codeword = list(seq)
            codeword.append(s)
            t.append([codeword, seqProb * dist[i]])
        for i in range(0, len(seq)):
            if i == 0:
                cur = tunstall_tree[seq[i]]
            else:
                cur = cur[2][seq[i]]
        son_node = []
        for i, s in enumerate(alphabet):
            son_node.append([s, seqProb * dist[i], None, None])
        cur[2] = son_node
        del t[ind]

    codebook, _ = generate_codebook([-1, 1.0, tunstall_tree, None], [], 0)

    return tunstall_tree, codebook


def tunstall_encoding(data, tunstall_tree, n):
    n_data = len(data)
    for i in range(0, n):
        data = np.append(data, np.array([0]), axis=0)
    tunstall_codes = []

    cur = None
    for i in range(0, len(data)):
        val = int(data[i])
        if cur == None:
            cur = tunstall_tree[val]
        else:
            cur = cur[2][val]
        if cur[2] == None:
            tunstall_codes.append(cur[3])
            cur = None
            if i >= n_data:
                break

    return tunstall_codes

def tunstall_decoding(codes, codebook, n):
    values = [0] * n
    cnt = 0
    for i in range(0, len(codes)):
        id = codes[i]
        for j in range(0, len(codebook[id])):
            values[cnt] = codebook[id][j]
            cnt = cnt + 1
    return values

def calc_tunstall_codes_size(data, n):
    values, counts = np.unique(data, return_counts=True)
    dicts = {}
    inv_dicts = {}
    for i in range(0, len(values)):
        dicts[values[i]] = i
        inv_dicts[i] = values[i]

    data = data.reshape(-1)

    if len(counts) == 1:
        return 0.0, data

    n_data = len(data)
    for i in range(0, len(data)):
        data[i] = dicts[data[i]]
    alphabet = [i for i in range(0, len(values))]

    tunstall_tree, codebook = create_tunstall_tree(alphabet, counts / counts.sum(), n)
    tunstall_codes = tunstall_encoding(data, tunstall_tree, n)
    decoded_values = tunstall_decoding(tunstall_codes, codebook, n_data + n)
    for i in range(0, len(decoded_values)):
        decoded_values[i] = inv_dicts[decoded_values[i]]

    return len(tunstall_codes) * n / len(data), decoded_values

def calc_tunstall_codes_size_fast(data, n):
    values, counts = np.unique(data, return_counts=True)
    if len(counts) == 1:
        return 0.0

    dicts = {}
    inv_dicts = {}
    for i in range(0, len(values)):
        dicts[values[i]] = i
        inv_dicts[i] = values[i]

    data = data.reshape(-1)

    if len(counts) == 1:
        return 0.0, data

    n_data = len(data)
    for i in range(0, len(data)):
        data[i] = dicts[data[i]]
    alphabet = [i for i in range(0, len(values))]

    return ave_tunstall_code_size(alphabet, counts / counts.sum(), n)


def preprocessing(data):
    values, counts = np.unique(data, return_counts=True)
    counts = counts / counts.sum()
    pairs_v_p = [(values[i], counts[i]) for i in range(0, len(values))]
    pairs_v_p.sort(key=lambda p:p[1], reverse=True)
    dicts = {}
    inv_dicts = {}

    for i in range(0, len(pairs_v_p)):
        dicts[pairs_v_p[i][0]] = i
        inv_dicts[i] = pairs_v_p[i][0]

    data = np.array(data.reshape(-1))
    for i in range(0, len(data)):
        data[i] = dicts[data[i]]

    return data, inv_dicts


def remove_zero_bit_filters(data, bit_allocations):
    [nfilters, ndepth, nheight, nwidth] = data.size()
    np_data = data.cpu().numpy().reshape(-1)

    np_data_zero_removed = [0] * (nfilters * ndepth * nheight * nwidth)
    filter_size = ndepth * nheight * nwidth
    cnt = 0
    for i in range(0, nfilters):
        if (bit_allocations[i] != 0):
            np_data_zero_removed[(cnt * filter_size): (cnt * filter_size + filter_size)] = \
                    np_data[(i * filter_size): (i * filter_size + filter_size)]
            cnt = cnt + 1

    return np.array(np_data_zero_removed[0:(cnt*filter_size)])

def cal_multistage_tunstall_codes_size(tensor_data, nbittree, nstage, evaluate=False):
    # original_data = remove_zero_bit_filters(tensor_data, bits)
    original_data = tensor_data.cpu().numpy().reshape(-1)
    data, inv_dicts = preprocessing(original_data)
    #[nfilters, ndepth, nheight, nwidth] = tensor_data.size()
    #n_data = nfilters * ndepth * nheight * nwidth
    n_data = tensor_data.numel()
    rate = 1e8
    K_R_stages = [0] * nstage

    for s in range(0, nstage):
        K = np.amax(data) + 1
        if (K % 2) == 0:
            K_R = K / 2 + 1
        else:
            K_R = (K + 1) / 2 + 1
        K_R_stages[s] = K_R
        d_data = [0] * (2 * len(data))
        cnt = 0
        for i in range(len(data)):
            if data[i] < K_R - 1:
                d_data[cnt] = data[i]
                cnt = cnt + 1
            else:
                d_data[cnt] = (K_R - 1)
                cnt = cnt + 1
                d_data[cnt] = data[i] - (K_R - 1)
                cnt = cnt + 1
        data = d_data[0:cnt]
        data = np.array(data)
        _rate, decoded_values = calc_tunstall_codes_size(data, nbittree)
        _rate = _rate * len(data) / n_data
        if _rate < rate:
            rate = _rate

    if evaluate == True:
        for s in range(0, nstage):
            _decoded_values = [0] * len(decoded_values)
            id = 0
            cnt = 0
            while True:
                if (decoded_values[id] == (K_R_stages[nstage - 1 - s] - 1)):
                    _decoded_values[cnt] = decoded_values[id] + decoded_values[id + 1]
                    cnt = cnt + 1
                    id = id + 2
                else:
                    _decoded_values[cnt] = decoded_values[id]
                    cnt = cnt + 1
                    id = id + 1
                if id >= len(decoded_values):
                    break
            decoded_values = _decoded_values[0:cnt]

        for i in range(0, len(decoded_values)):
            decoded_values[i] = inv_dicts[decoded_values[i]]

        diff = original_data - decoded_values[0:len(original_data)]
        if (diff.sum() == 0):
            print('multistage tunstall coding success')
        else:
            print('multistage tunstall coding error %f' % diff.sum())

    #print('process tunstal coding %d' % (len(data)))
    return rate

def cal_multistage_tunstall_codes_size_fast(tensor_data, nbittree, nstage, num_samples=20000, evaluate=False):
    # original_data = remove_zero_bit_filters(tensor_data, bits)
    original_data = tensor_data.cpu().numpy().reshape(-1)
    if len(original_data) > num_samples:
        original_data = random.sample(list(original_data), num_samples)
        original_data = np.array(original_data)
    n_data = len(original_data)

    data, inv_dicts = preprocessing(original_data)
    #[nfilters, ndepth, nheight, nwidth] = tensor_data.size()
    #n_data = nfilters * ndepth * nheight * nwidth

    rate = 1e8
    K_R_stages = [0] * nstage

    for s in range(0, nstage):
        K = np.amax(data) + 1
        if (K % 2) == 0:
            K_R = K / 2 + 1
        else:
            K_R = (K + 1) / 2 + 1
        K_R_stages[s] = K_R
        d_data = [0] * (2 * len(data))
        cnt = 0
        for i in range(len(data)):
            if data[i] < K_R - 1:
                d_data[cnt] = data[i]
                cnt = cnt + 1
            else:
                d_data[cnt] = (K_R - 1)
                cnt = cnt + 1
                d_data[cnt] = data[i] - (K_R - 1)
                cnt = cnt + 1
        data = d_data[0:cnt]
        data = np.array(data)
        _rate = calc_tunstall_codes_size_fast(data, nbittree)
        #print('testing')
        #print(_rate)
        #print(len(data))
        #print(n_data)

        _rate = _rate * len(data) / n_data

        if _rate < rate:
            rate = _rate

    return rate

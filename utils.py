import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math as m
from copy import deepcopy
import itertools
from tqdm import tqdm

import networkx as nx
from networkx import gnp_random_graph, gnm_random_graph, random_regular_graph, grid_2d_graph, random_tree, star_graph

from discreteMarkovChain import markovChain


def generate_gnp(n, p, verbose=True, draw=False, save=False, seed=None):
    G = gnp_random_graph(n , p, seed=seed)

    if verbose==True:
        print('{} edges'.format(len(G.edges())))
    if draw==True:
        plt.figure(figsize=(4,3))
        nx.draw(G)
        if save==True:
            plt.savefig('/Users/philchen/Desktop/Gnp.png',dpi=100)
        plt.show()
    return G


def generate_gnm(n, m, verbose=True, draw=False, save=False, seed=None):
    G = gnm_random_graph(n, m, seed=seed)

    if verbose==True:
        print('{} edges'.format(len(G.edges())))
    if draw==True:
        plt.figure(figsize=(4,3))
        nx.draw(G)
        if save==True:
            plt.savefig('/Users/philchen/Desktop/Gnm.png',dpi=100)
        plt.show()
    return G


def generate_gd(n, D, verbose=True, draw=False, save=False, seed=None):
    G = random_regular_graph(D, n, seed=seed)

    if verbose==True:
        print('{} edges'.format(len(G.edges())))
    if draw==True:
        plt.figure(figsize=(4,3))
        nx.draw(G)
        if save==True:
            plt.savefig('/Users/philchen/Desktop/GD.png',dpi=100)
        plt.show()
    return G


def generate_lattice(row, col, verbose=True, draw=False, save=False):
    G = grid_2d_graph(m=row, n=col)
    lattice = nx.Graph()
    for e in G.edges():
        i = e[0][0]*col + e[0][1]
        j = e[1][0]*col + e[1][1]
        lattice.add_edge(i, j)

    if verbose==True:
        print('{} edges'.format(len(lattice.edges())))
    if draw==True:    
        plt.figure(figsize=(4,3))
        nx.draw(lattice)
        if save==True:
            plt.savefig('/Users/philchen/Desktop/lattice.png',dpi=100)
        plt.show()
    return lattice


def generate_d_regular_bipartitle(n, d, verbose=True, draw=False, save=False):
    assert d <= n
    G = nx.Graph()
    for i in range(n):
        for j in range(d):
            if i+n+j < 2*n:
                G.add_edge(i, i+n+j)
            else:
                G.add_edge(i, i+j)
    
    if verbose==True:
        print('{} edges'.format(len(G.edges())))
    if draw==True:    
        plt.figure(figsize=(4,3))
        nx.draw(G)
        if save==True:
            plt.savefig('/Users/philchen/Desktop/d_regular_bipartitle.png',dpi=100)
        plt.show()
    
    return G


def set_spin(beta, gamma, lamda, Ising=False):
    if Ising==True:
        assert beta == gamma
    else:
        assert beta*gamma > 1
        assert beta < gamma
    A = np.array( [[beta, 1], [1, gamma]] ).astype(float)
    b = np.array( [lamda, 1] ).astype(float)
    
    return A, b


def compute_weight(G, A, b, cfg):
    weight = b[0]**(np.sum(cfg==0))
    for e in G.edges():
        weight *= A[ cfg[e[0]], cfg[e[1]] ]
    return weight


def bitlist_to_int(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def int_to_bitlist(x, n):
    int_str = bin(x)[2:]
    int_str = '0'*(n-len(int_str))+int_str
    return np.array(list(int_str), dtype=np.int8)


def flip_one_bit(cfg, idx):
    cfg_new = deepcopy(cfg)
    cfg_new[idx] = cfg_new[idx] ^ 1
    return cfg_new


def get_pi(G, A, b, verbose=False):
    n = len(G.nodes)
    N = 2**n
    cfg_list = [[0,1]]*n
    weights = []

    for cfg in itertools.product(*cfg_list):
        cfg = np.array(cfg)
        weight = compute_weight(G, A, b, cfg)
        weights.append(weight)

    weights = np.array(weights)
    assert weights.size == N
    assert weights.dtype == float
    
    pi = weights / np.sum(weights)
    if verbose:
        try:
            assert abs(np.sum(pi)-1) < pi.min()*1e-1
        except:
            print('err={:.3E}  min={:.3E}'.format(abs(np.sum(pi)-1), pi.min()*1e-1))
            # raise
    
    return pi


def get_transition_matrix(G, A, b, verbose=False):
    n = len(G.nodes)
    N = 2**n
    cfg_list = [[0,1]]*n
    Omega = []
    weights = []

    for cfg in itertools.product(*cfg_list):
        cfg = np.array(cfg)
        weight = compute_weight(G, A, b, cfg)
        weights.append(weight)
        Omega.append(cfg)

    weights = np.array(weights)
    Omega = np.array(Omega)
    assert weights.size == N
    assert weights.dtype == float
    
    pi = weights / np.sum(weights)
    if verbose:
        try:
            assert abs(np.sum(pi)-1) < pi.min()*1e-1
        except:
            print('err={:.3E}  min={:.3E}'.format(abs(np.sum(pi)-1), pi.min()*1e-1))
            # raise
    
    P = np.zeros((N, N))

    for idx_a in range(N):
        cfg_a = Omega[idx_a]
        assert idx_a == bitlist_to_int(cfg_a)
        weight_a = weights[idx_a]
        # pick a vetex u.a.r.
        for i in range(n):
            cfg_b = flip_one_bit(cfg_a, i)
            idx_b = bitlist_to_int(cfg_b)
            weight_b = weights[idx_b]
            P[idx_a, idx_b] = weight_b/(weight_a+weight_b)/n
            P[idx_a, idx_a] += 1./n - P[idx_a, idx_b]
    
    return P, pi


def get_t_rel(P):
    E, V = LA.eig(P)
    E_abs = np.sort(abs(E))[::-1]
    asp = 1 - E_abs[1]
    t_rel = 1/asp
    
    return t_rel


def get_t_mix(P, pi, epsilon):
    assert P.dtype == float
    assert pi.dtype == float
    
    N = P.shape[0]
    distributions = np.identity(N)
    
    t_mix=0; d_t=1
    while d_t > epsilon:
        TVs = []
        for i in range(N):
            distributions[i] = np.matmul(distributions[i], P)
            TVs.append(total_variation(pi,distributions[i])) 
        d_t = np.max(TVs)
        t_mix += 1
    
    return t_mix


def get_t_mix_by_sampling(G, A, b, pi, epsilon, interval, init_cfg=None, seed=None):
    n = int(np.log2(pi.size))
    if init_cfg==None:
        np.random.seed(seed)
        cfg = np.random.randint(2, size=n, dtype=np.int8)
    else:
        cfg = int_to_bitlist(init_cfg, n)
    
    samples = np.zeros(2**n, dtype=int)
    samples[bitlist_to_int(cfg)] += 1

    t_mix = 0; d_t = 1
    while d_t > epsilon:
        for i in range(interval):
            cfg = sample(G, A, b, cfg)
            samples[bitlist_to_int(cfg)] += 1
        dist = samples/np.sum(samples)
        d_t = total_variation(dist, pi)
        t_mix += 1
    
    return t_mix*interval


def total_variation(mu, nu):
    return sum(abs(mu-nu))/2


def get_d_t(P, pi, timesteps):
    assert P.dtype == float
    assert pi.dtype == float
 
    N = P.shape[0]
    Total_Variation = []
    distribution = np.identity(N)
    
    Total_Variation.append( [total_variation(pi, distribution[i]) for i in range(N)] )
    for t in range(timesteps):
        distribution = np.matmul(distribution, P)
        Total_Variation.append( [total_variation(pi, distribution[i]) for i in range(N)] )

    Total_Variation = np.array(Total_Variation)
    d_t = np.max(Total_Variation, axis=1)
    
    return d_t


def get_lambda_c(beta, gamma, Ising=False, verbose=False):
    if Ising==True:
        assert beta == gamma
    else:
        assert beta*gamma > 1
        assert beta < gamma
    
    Delta_c = 1. + 2/(m.sqrt(beta*gamma)-1)
    lamda_c = (gamma/beta)**((Delta_c+1)/2)
    lamda_c_int = (gamma/beta)**((m.ceil(Delta_c)+1)/2)
    lamda_c_int_p = (gamma/beta)**((m.floor(Delta_c)+2)/2)
    if verbose:
        print('lambda_c = {:.5f}'.format(lamda_c))
        print('lambda_c_int = {:.5f}'.format(lamda_c_int))
        print('lambda_c_int_p = {:.5f}'.format(lamda_c_int_p))
        print('Delta_c = {:.5f}'.format(Delta_c))
        print('Delta_c+1 = {:.5f}'.format(Delta_c+1))

    
    return lamda_c, lamda_c_int_p, Delta_c


def get_delta_c(beta, gamma):
    tmp = m.sqrt(beta*gamma)
    assert tmp>1
    return (tmp+1)/(tmp-1)


def sample(G, A, b, cfg):
    n = len(G.nodes)
    cfg_a = cfg
    
    # select a node u.a.r.
    node_idx = np.random.randint(n)
    cfg_b = flip_one_bit(cfg_a, node_idx)
    
    neighbors = np.zeros(2, dtype=int)
    for neighbor in G.neighbors(n=node_idx):
        neighbors[cfg[neighbor]] += 1
    
    a_val = cfg_a[node_idx]
    b_val = cfg_b[node_idx]
    ab_ratio = (b[a_val]/b[b_val])
    ab_ratio *= (A[a_val,0]/A[b_val,0])**neighbors[0]
    ab_ratio *= (A[a_val,1]/A[b_val,1])**neighbors[1]
    
    cfgs = [cfg_a, cfg_b]
    probs = np.zeros(2)
    probs[1] = 1/(ab_ratio+1)
    probs[0] = 1 - probs[1]
    
    # sample
    idx = np.random.choice(2, size=1, p=probs)
    
    return cfgs[idx[0]]


def get_weights(G, A, b):
    n = len(G.nodes)
    cfg_list = [[0,1]]*n
    weights = {}

    for cfg in itertools.product(*cfg_list):
        cfg = np.array(cfg)
        weight = compute_weight(G, A, b, cfg)
        key = np.sum(cfg)
        if key in weights.keys():
            weights[key].append(weight)
        else:
            weights[key] = [weight]
    
    return weights


def get_group_weights(G, A, b):
    weights = get_weights(G, A, b)
    group_weights = []
    for key, value in weights.items():
        group_weights.append( np.sum(value) )
    group_weights = np.array(group_weights)

    return group_weights


def get_grouped_pi(G, A, b):
    weights = get_weights(G, A, b)
    res = []
    for key, value in weights.items():
        res.extend(value)
    res = np.array(res)
    grouped_pi = res / np.sum(res)
    
    return grouped_pi


def space_travel(G, A, b, start, alpha):
    n = len(G.nodes)
    cfg = start - np.zeros(n, dtype=int)

    step = 0
    while abs(np.sum(cfg)-n*start) < int(alpha*n):
        cfg = sample(G, A, b, cfg)
        step += 1
    
    return step


def Cnr(n, r):
    assert 0 <= r <= n
    res = m.factorial(n) / m.factorial(r) / m.factorial(n-r)
    
    return int(res)


def sample_cnt(G, A, b, init_cfg, iters):
    n = len(G.nodes())
    samples = np.zeros(n+1, dtype=int)
    samples[init_cfg.sum()] += 1
    
    cfg = init_cfg
    for i in tqdm(range(iters)):
        cfg = sample(G, A, b, cfg)
        idx = cfg.sum()
        samples[idx] += 1
    return samples
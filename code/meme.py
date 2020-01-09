#!/usr/bin/env python3

''' Performs EM on the specified set of sequences to identify motifs

Arguments:
    -f: set of sequences to identify a motif in
    -W: the length of the desired motif to find
    -init: the initialization method for theta_1
    -backg: the initialization method for theta_2
Outputs:
    - Iterations to EM convergence
    - Expected log-likelihood of mixture model
    - Total time to train
    - a list of start positions for each sequence, along with the corresponding
      motif identified
    - the consensus motif (based on majority base at each position)

Example Usage:
    python meme.py -f data/MA0006.1/MA0006.1-motif1.sites -W 6 -init plain -backg unif
'''

import argparse
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import time

BASE_TO_IND = dict(zip(['A','C','G','T'], range(4)))

''' Returns the majority vote (consensus) motif sequence.
From course staff.  
Arguments:
    motif_sequences: list of all motif instances (list of strings)
Returns:
    majority sequence: consensus motif sequence (string)
'''
def majority(motif_sequences):
    base_ordering = {c: i for i, c in enumerate("ACGT")}
    freqs = np.zeros((len(motif_sequences[0]), 4))
    for m_s in motif_sequences:
        for i, b in enumerate(m_s): freqs[i, base_ordering[b]] += 1
    return ''.join(['ACGT'[np.argmax(row)] for row in freqs])

''' Reads in the sequences from the motif files.
From course staff.  
Arguments:
    filename: which filename to read sequences from
Returns:
    output: list of sequences in motif file
'''
def read_fasta(filename):
    with open(filename, "r") as f:
        output = []
        s = ""
        for l in f.readlines():
            if l.strip()[0] == ">":
                # skip the line that begins with ">"
                if s == "": continue
                output.append(s.upper())
                s = ""
            # keep appending to the current sequence when the line doesn't begin
            # with ">"
            else: s += l.strip()
        output.append(s.upper())
        return output

def preprocess(sequences, W):
    X = []
    #index_map1 = {}
    # index_map1 was originally used for the renormalization (_normalize) of
    # posterior probabilities over all W-mers
    index_map2 = {}
    i = 0
    for m, Y in enumerate(sequences):
        l_m = len(Y)
        for p in range(l_m - W + 1):
            end_p = W + p - 1 + 1
            subsequence = list(Y[p:end_p])
            X.append(subsequence)
            #index_map1[(m,p)] = i
            index_map2[i] = (m,p)
            i += 1
    X = np.array(X)
    return X, index_map2

def background_probs(sequences):
    frequencies = Counter([base for seq in sequences for base in seq])
    total_freq = sum(frequencies.values())
    F_backg = np.zeros(4)
    for base, base_freq in frequencies.items():
        index = BASE_TO_IND[base]
        F_backg[index] = base_freq/total_freq
    return F_backg

def _normalize(Z, sequences, index_map):
    N = len(sequences)
    for m in range(N):
        l_m = len(sequences[m])
        for p in range(l_m - 5 + 1):
            i = index_map[(m,p)]
            end_i = i + 5 - 1 + 1
            motif_probs = Z[i:end_i,0]
            Z[i:end_i,0] = motif_probs/np.sum(motif_probs)
    return Z

def likelihood_subseq(E, F_motif, F_backg, X):
    n, W = X.shape
    for i in range(n):
        motif_prob = 1
        backg_prob = 1
        for w in range(W):
            for a in range(4):
                is_base_a = BASE_TO_IND[X[i,w]] == a
                motif_prob *= F_motif[w,a]**is_base_a
                backg_prob *= F_backg[a]**is_base_a
        E[i,0] = motif_prob
        E[i,1] = backg_prob
    return E

def expectation(E, Z, lambdas):
    n, _ = Z.shape
    for i in range(n):
        normalizer = np.sum(E[i,:]*lambdas)
        for j in range(2):
            Z[i,j] = E[i,j]*lambdas[j]/normalizer
    return Z

def _expected_counts(Z, X):
    n, W = X.shape
    C_motif = np.zeros((W,4))
    C_backg = np.zeros(4)

    for a in range(4):
        backg_exp_count = 0
        for w in range(W):
            motif_exp_count = 0
            for i in range(n):
                is_base_a = BASE_TO_IND[X[i,w]] == a
                motif_exp_count += Z[i,0]*is_base_a
                backg_exp_count += Z[i,1]*is_base_a
            C_motif[w,a] = motif_exp_count
        C_backg[a] = backg_exp_count

    return C_motif, C_backg

def maximization(F_motif, F_backg, Z, betas, X):
    _, W = X.shape
    C_motif, C_backg = _expected_counts(Z, X)

    for w in range(W):
        motif_normalizer = np.sum(C_motif[w,:] + betas)
        F_motif[w,:] = (C_motif[w,:] + betas)/motif_normalizer
    
    backg_normalizer = np.sum(C_backg + betas)
    F_backg = (C_backg + betas)/backg_normalizer

    lambdas = np.mean(Z, axis=0)
    return F_motif, F_backg, lambdas

def exp_log_likelihood(E, lambdas, Z):
    n, _ = E.shape
    lambdas = np.tile(lambdas, (n,1))
    return np.sum(Z*np.log(E)) + np.sum(Z*np.log(lambdas))

def em(X, F_motif, F_backg, lambdas, betas, stop_diff=10**-2, maxiters=100):
    n, _ = X.shape
    E = np.zeros((n,2)) # just need a placeholder
    Z = np.zeros((n,2)) # just need a placeholder

    iteration = 0
    exp_logLs = []

    E = likelihood_subseq(E, F_motif, F_backg, X)
    Z = expectation(E, Z, lambdas)
    exp_logL = exp_log_likelihood(E, lambdas, Z)
    diff = stop_diff + 1
    exp_logLs.append(exp_logL)
    iteration += 1

    while diff > stop_diff and iteration <= maxiters:
        F_motif, F_backg, lambdas = maximization(F_motif, F_backg, Z, betas, X)
        E = likelihood_subseq(E, F_motif, F_backg, X)
        Z = expectation(E, Z, lambdas)
        exp_logL = exp_log_likelihood(E, lambdas, Z)
        diff = exp_logL - exp_logLs[-1]
        exp_logLs.append(exp_logL)
        iteration += 1
    
    return Z, exp_logL, exp_logLs, F_motif

def plain_initialized_MEME(X, F_backg, betas, filename, N):
    _, W = X.shape
    F_motif = np.tile(np.array([0.25,0.25,0.25,0.25]), (W,1))
    lambdas = np.array([1/N, 1-(1/N)])

    Z, exp_logL, exp_logLs, F_motif = em(X, F_motif, F_backg, lambdas, betas)

    rounded_lamdba1 = np.round(lambdas[0], 4)
    print(f"Iterations to converge : {len(exp_logLs)}")
    print(f"Expected log-likelihood: {exp_logL}")
    print(f"Pre-determined lambda1 : {rounded_lamdba1}")
    print()

    plt.plot(exp_logLs, label=f"lambda1 = {rounded_lamdba1}")
    plt.xlabel("Iteration")
    plt.ylabel("Expected log-likelihood")
    filename = filename.split("/")[-1]
    plt.title(f"{filename} - Plain Initialization")
    plt.legend(loc='lower right')
    #plt.savefig(f"figures/EM {filename} - Plain.png")

    return Z, F_motif

def _lambda1_line(W, N, n):
    lambda1s = []
    start = (N**0.5)/n
    end = 1/(2*W)
    lambda1 = start
    while lambda1 < end:
        lambda1s.append(lambda1)
        lambda1 *= 2
    
    lambda1s.append(1/N)
    return np.array(lambda1s)

def min_divergence(x):
    W = len(x)
    m = 0.40 # hardcoded for gamma=0.05
    F_motif = ((1-m)/3)*np.ones((W,4))
    for w in range(W):
        for a in range(4):
            if BASE_TO_IND[x[w]] == a:
                F_motif[w,a] = m
    return F_motif

def smart_initialized_MEME(X, F_backg, betas, filename, N):
    n, W = X.shape
    alpha = 0.9

    best_Z = None
    best_exp_logL = -np.inf
    best_iters = 0
    best_lambda1 = 0
    best_F_motif = None

    lambda1s = _lambda1_line(W, N, n)

    for lambda1 in lambda1s:
        lambdas = np.array([lambda1, 1-lambda1])
        best_lambda1_exp_logL = -np.inf
        best_lambda1_F_motif = None

        Q = int(np.log(1-alpha)/np.log(1-lambda1))
        Q = min((n, Q))
        draws = np.random.choice(n, Q, replace=False)

        for q in range(Q):
            draw = draws[q]
            x = X[draw,:]
            # if np.all(np.char.equal(x, ['C','G','C','G','T','G'])):
            #     print("Motif found")
            F_motif_initial = min_divergence(x)

            _, exp_logL, _, _ = em(X, F_motif_initial, F_backg, lambdas, betas, maxiters=1)
            #print(exp_logL)

            if exp_logL > best_lambda1_exp_logL:
                best_lambda1_exp_logL = exp_logL
                best_lambda1_F_motif = F_motif_initial

        Z, exp_logL, exp_logLs, F_motif = em(X, best_lambda1_F_motif, F_backg, lambdas, betas)
        if exp_logL > best_exp_logL:
            best_Z = Z
            best_exp_logL = exp_logL
            best_iters = len(exp_logLs)
            best_lambda1 = lambda1
            best_F_motif = F_motif

        rounded_lamdba1 = np.round(lambda1, 4)
        plt.plot(exp_logLs, label=f"lambda1 = {rounded_lamdba1}")
    
    print(f"Iterations to converge : {best_iters}")
    print(f"Expected log-likelihood: {exp_logL}")
    print(f"Selected lambda1       : {np.round(best_lambda1, 4)}")
    print()

    plt.xlabel("Iteration")
    plt.ylabel("Expected log-likelihood")
    filename = filename.split("/")[-1]
    plt.title(f"{filename} - Smart Initialization")
    plt.legend(loc='lower right')
    #plt.savefig(f"figures/EM {filename} - Smart.png")

    return best_Z, best_F_motif

def main():
    parser = argparse.ArgumentParser(description='Estimate the start positions and motif model via EM.')
    parser.add_argument('-f', action="store", dest="f", type=str, default='data/MA0006.1/MA0006.1-motif1.sites')
    parser.add_argument('-W', action="store", dest="W", type=int, default=6)
    parser.add_argument('-init', action="store", dest="init", type=str, default='plain')
    parser.add_argument('-backg', action="store", dest="backg", type=str, default='unif')

    args = parser.parse_args()
    print("Reading FASTA.")
    filename = args.f
    sequences = read_fasta(filename)
    N = len(sequences)
    W = args.W
    ''' Must define a beta value. '''
    beta = 0.1
    top_k = 10

    print("Pre-processing sequences and storing mapping.")
    print()
    X, index_map = preprocess(sequences, W)
    betas = np.repeat(beta, 4)

    if args.backg == "unif":
        F_backg = np.ones(4)/4
    else:
        F_backg = background_probs(sequences)
    
    print("Running MEME.")
    if args.init == "plain":
        start = time.time()
        Z, _ = plain_initialized_MEME(X, F_backg, betas, filename, N)
        end = time.time()
    else:
        start = time.time()
        Z, _ = smart_initialized_MEME(X, F_backg, betas, filename, N)
        end = time.time()
    
    print(f"MEME and plotting took {np.round(end-start,1)} seconds.")
    print()
    
    #print(F_motif)

    Z_motif = Z[:,0]
    motif_indices = np.argsort(-Z_motif)[:top_k]
    motif_sequences = []
    for ind in motif_indices:
        motif = ''.join(X[ind,:])
        motif_sequences.append(motif)
        m, p = index_map[ind]
        # zero to one indexing
        add_0_m = '0' if (m+1) < 10 else ''
        add_0_p = '0' if (p+1) < 10 else ''
        print(f"Sequence {add_0_m}{m+1} position {add_0_p}{p+1}: {motif}")
    
    concensus = majority(motif_sequences)
    print()
    print(f"        Concensus motif: {concensus}")

if __name__ == '__main__':
    main()

# python meme.py -f data/MA0006.1/MA0006.1-motif1.sites -W 6 -init plain -backg unif
# python meme.py -f data/MA0006.1/MA0006.1-motif1.sites -W 6 -init smart -backg unif
# python meme.py -f data/MA0006.1/MA0006.1-motif2.sites -W 6 -init plain -backg unif
# python meme.py -f data/MA0006.1/MA0006.1-motif2.sites -W 6 -init smart -backg unif

# python meme.py -f data/MA0259.1/MA0259.1-motif1.sites -W 8 -init plain -backg unif
# python meme.py -f data/MA0259.1/MA0259.1-motif1.sites -W 8 -init smart -backg unif
# python meme.py -f data/MA0259.1/MA0259.1-motif2.sites -W 8 -init plain -backg unif
# python meme.py -f data/MA0259.1/MA0259.1-motif2.sites -W 8 -init smart -backg unif
# set gamma to 0.3 in last command. 

# python meme.py -f data/MA0006.1/MA0006.1-motif1.sites -W 6 -init plain -backg estimate
# python meme.py -f data/MA0006.1/MA0006.1-motif1.sites -W 6 -init smart -backg estimate
# python meme.py -f data/MA0006.1/MA0006.1-motif2.sites -W 6 -init plain -backg estimate
# python meme.py -f data/MA0006.1/MA0006.1-motif2.sites -W 6 -init smart -backg estimate

# python meme.py -f data/MA0259.1/MA0259.1-motif1.sites -W 8 -init plain -backg estimate
# python meme.py -f data/MA0259.1/MA0259.1-motif1.sites -W 8 -init smart -backg estimate
# python meme.py -f data/MA0259.1/MA0259.1-motif2.sites -W 8 -init plain -backg estimate
# python meme.py -f data/MA0259.1/MA0259.1-motif2.sites -W 8 -init smart -backg estimate
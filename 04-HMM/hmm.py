# Author: Kaituo Xu, Fan Yu

# need numpy
import numpy as np


def forward_algorithm(O, HMM_model):
    """HMM Forward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    npi = np.array(pi)
    nA = np.array(A)
    nB = np.array(B)
    nO = np.array(O)

    # First step:initial
    alpha = np.zeros((1, N))
    alpha = npi * nB[:, nO[0]]

    # Second step:iteration
    for i in range(1, T):
        alpha = alpha.dot(nA) * nB[:, nO[i]]

    # Final step:sum
    prob = np.sum(alpha)

    # End Assignment
    return prob


def backward_algorithm(O, HMM_model):
    """HMM Backward Algorithm.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Return:
        prob: the probability of HMM_model generating O.
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    prob = 0.0
    # Begin Assignment

    # Put Your Code Here
    npi = np.array(pi)
    nA = np.array(A)
    nB = np.array(B)
    nO = np.array(O)
    # First step:initial
    beta = np.zeros((1, N))
    beta = 1

    # Second step:iteration
    for i in range(T-2, -1, -1):
        beta = nA.dot((beta * nB[:, nO[i+1]]).T)

    # Final step:sum
    prob = np.sum(npi * nB[:, nO[0]] * beta)

    # End Assignment
    return prob


def Viterbi_algorithm(O, HMM_model):
    """Viterbi decoding.
    Args:
        O: (o1, o2, ..., oT), observations
        HMM_model: (pi, A, B), (init state prob, transition prob, emitting prob)
    Returns:
        best_prob: the probability of the best state sequence
        best_path: the best state sequence
    """
    pi, A, B = HMM_model
    T = len(O)
    N = len(pi)
    best_prob, best_path = 0.0, []
    # Begin Assignment
    npi = np.array(pi)
    nA = np.array(A)
    nB = np.array(B)
    nO = np.array(O)
    # Put Your Code Here
    # First step:initial
    delta = np.zeros((T, N))
    path = np.zeros((T, N), dtype=int)
    mid = np.zeros((N, N))
    delta[0] = npi * nB[:, nO[0]]

    # Second step:iteration
    for i in range(1, T):
        mid = delta[i-1].reshape(N, 1) * nA
        delta[i] = np.max(mid, axis=0)*nB[:, nO[i]]
        path[i] = np.argmax(mid, axis=0)

    # Third step:finish
    best_prob = np.max(delta[T-1])
    j = np.argmax(delta[T-1])
    best_path.append(j)

    # Final step:
    for i in range(T-1, 0, -1):
        j = path[i][j]
        best_path.append(j)

    best_path.reverse()
    # End Assignment
    return best_prob, best_path


if __name__ == "__main__":
    color2id = {"RED": 0, "WHITE": 1}
    # model parameters
    pi = [0.2, 0.4, 0.4]
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    # input
    observations = (0, 1, 0)
    HMM_model = (pi, A, B)
    # process
    observ_prob_forward = forward_algorithm(observations, HMM_model)
    print(observ_prob_forward)

    observ_prob_backward = backward_algorithm(observations, HMM_model)
    print(observ_prob_backward)

    best_prob, best_path = Viterbi_algorithm(observations, HMM_model)
    print(best_prob, best_path)

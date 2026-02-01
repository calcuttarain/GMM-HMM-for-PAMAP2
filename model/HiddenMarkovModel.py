import numpy as np
import pickle
from .GaussianMixtureModel import GaussianMixtureModel

class HiddenMarkovModel:
    def __init__(self, num_states, num_components_mixtures, data_dim):
        self.N = num_states
        self.M_vec = num_components_mixtures
        self.D = data_dim
        
        self.TOL = 1e-300
        
        # initialize the transition matrix
        self.A = np.random.rand(self.N, self.N)
        self.A /= self.A.sum(axis=1, keepdims=True)
        
        # initialize initial state distribution
        self.pi = np.random.rand(self.N)
        self.pi /= self.pi.sum()
        
        # initialize the gaussian mixtures for each state
        self.states = [GaussianMixtureModel(self.M_vec[i], self.D) for i in range(self.N)]

    def _forward_procedure(self, B):
        T = B.shape[1]
        alpha = np.zeros((self.N, T))
        scales = np.zeros(T) 
        
        alpha[:, 0] = self.pi * B[:, 0]
        
        scales[0] = 1.0 / (np.sum(alpha[:, 0]) + self.TOL)
        alpha[:, 0] *= scales[0]
        
        for t in range(1, T):
            term1 = alpha[:, t-1] @ self.A  
            alpha[:, t] = term1 * B[:, t]
            
            scales[t] = 1.0 / (np.sum(alpha[:, t]) + self.TOL)
            alpha[:, t] *= scales[t]
            
        return alpha, scales

    def _backward_procedure(self, B, scales):
        T = B.shape[1]
        beta = np.zeros((self.N, T))
        
        beta[:, T-1] = 1.0 
        beta[:, T-1] *= scales[T-1]
        
        for t in range(T-2, -1, -1):
            term1 = self.A * B[:, t+1] * beta[:, t+1] 
            beta[:, t] = np.sum(term1, axis=1)
            beta[:, t] *= scales[t]
            
        return beta

    def train(self, observations, labels=None, method='unsupervised', n_iter=100, tol=1e-4):
        if method == 'supervised':
            self._train_supervised(observations, labels, n_iter_gmm = 30)
        else:
            self._train_baum_welch(observations, n_iter, tol)

    def _train_supervised(self, observations, labels, n_iter_gmm = 30):
        T = len(observations)
        
        counts_A = np.zeros((self.N, self.N))
        
        for t in range(T - 1):
            curr_s = labels[t]
            next_s = labels[t+1]
            counts_A[curr_s, next_s] += 1
            
        # Normalize rows + smoothing to avoid zero divisions
        self.A = counts_A / (counts_A.sum(axis=1, keepdims=True) + self.N)
        
        counts_pi = np.zeros(self.N)
        counts_pi[labels[0]] += 1 
        self.pi = counts_pi / (counts_pi.sum() + self.N)
        
        for s in range(self.N):
            state_data = observations[labels == s]
            
            indices = np.random.choice(len(state_data), self.states[s].M, replace=False)
            self.states[s].means = state_data[indices]
            
            for _ in range(n_iter_gmm):
                self.states[s].update_params(state_data, np.ones(len(state_data)))

    def _train_baum_welch(self, observations, n_iter=2000, tol=1e-4):
        T = observations.shape[0]
        prev_log_likelihood = -np.inf
        
        for iteration in range(n_iter):
            B = np.zeros((self.N, T))
            for n in range(self.N):
                B[n, :] = self.states[n].pdf(observations)
            
            alpha, scales = self._forward_procedure(B)
            beta = self._backward_procedure(B, scales)
            
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=0, keepdims=True) + self.TOL)
            
            xi = np.zeros((self.N, self.N, T-1))
            
            for t in range(T-1):
                num = alpha[:, t][:, np.newaxis] * self.A * B[:, t+1] * beta[:, t+1]
                xi[:, :, t] = num / (num.sum() + self.TOL)
            
            self.pi = gamma[:, 0]
            
            expected_transitions = np.sum(xi, axis=2) 
            expected_visits = np.sum(gamma[:, :-1], axis=1)[:, np.newaxis] 
            
            self.A = expected_transitions / (expected_visits + self.TOL)
            
            for n in range(self.N):
                state_posterior = gamma[n, :]
                self.states[n].update_params(observations, state_posterior)
                
            log_likelihood = -np.sum(np.log(scales + self.TOL))

            diff = log_likelihood - prev_log_likelihood
            
            if diff > 0:
                status = "INCREASING"
            elif diff < 0:
                status = "DECREASING"  
            else:
                status = "Converged"

            print(f"Iteration {iteration+1}: Log-Likelihood = {log_likelihood:.4f} (Change: {abs(diff):.4f} -> [{status}])")
            
            if abs(diff) < tol and iteration > 0:
                print("Converged.")
                break
                
            prev_log_likelihood = log_likelihood

    def predict_viterbi(self, observations):
        T = observations.shape[0]
        
        log_B = np.zeros((self.N, T))
        for n in range(self.N):
            log_B[n, :] = np.log(self.states[n].pdf(observations) + self.TOL)
            
        delta = np.zeros((self.N, T))
        psi = np.zeros((self.N, T), dtype=int)
        
        delta[:, 0] = np.log(self.pi + self.TOL) + log_B[:, 0]
        
        log_A = np.log(self.A + self.TOL)
        
        for t in range(1, T):
            for j in range(self.N):
                transitions_to_j = delta[:, t-1] + log_A[:, j]
                
                best_prev_score = np.max(transitions_to_j)
                best_prev_state = np.argmax(transitions_to_j)
                
                delta[j, t] = best_prev_score + log_B[j, t]
                
                psi[j, t] = best_prev_state
                
        path = np.zeros(T, dtype=int)
        path[T-1] = np.argmax(delta[:, T-1])
        
        for t in range(T-2, -1, -1):
            path[t] = psi[path[t+1], t+1]
            
        return path

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

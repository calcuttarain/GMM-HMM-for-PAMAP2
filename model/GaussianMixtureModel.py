import numpy as np

class GaussianMixtureModel:
    def __init__(self, num_components, mixture_dim):
        self.M = num_components
        self.D = mixture_dim
        
        self.weights = np.ones(self.M) / self.M
        self.means = np.zeros((self.M, self.D)) 
        self.Sigmas = np.array([np.eye(self.D) for _ in range(self.M)])
        
        self.TOL = 1e-300

    def pdf(self, observations):
        T, _ = observations.shape
        total_prob = np.zeros(T)
        
        for m in range(self.M):
            weight = self.weights[m]
            mean = self.means[m]
            Sigma = self.Sigmas[m]
            
            delta = observations - mean 
            
            try:
                inv_sigma = np.linalg.inv(Sigma)
                det_sigma = np.linalg.det(Sigma)
            except np.linalg.LinAlgError:
                inv_sigma = np.linalg.inv(Sigma + np.eye(self.D) * 1e-4)
                det_sigma = 1.0

            term1 = delta @ inv_sigma
            delta_2 = np.sum(term1 * delta, axis=1)

            if det_sigma <= 0: 
                det_sigma = self.TOL

            scalar = 1.0 / np.sqrt(((2 * np.pi) ** self.D) * det_sigma)
            normal = scalar * np.exp(-0.5 * delta_2)

            total_prob += weight * normal

        return np.maximum(total_prob, self.TOL)

    def update_params(self, observations, state_posterior):
        T, _ = observations.shape
        comp_likelihoods = np.zeros((T, self.M))
        
        for m in range(self.M):
            delta = observations - self.means[m]
            
            try:
                inv_sigma = np.linalg.inv(self.Sigmas[m])
                det_sigma = np.linalg.det(self.Sigmas[m])
            except np.linalg.LinAlgError:
                inv_sigma = np.eye(self.D)
                det_sigma = 1.0


            if det_sigma <= 0: 
                det_sigma = self.TOL
                
            term1 = delta @ inv_sigma
            delta_2 = np.sum(term1 * delta, axis=1)
            scalar = 1.0 / np.sqrt(((2 * np.pi) ** self.D) * det_sigma)
            
            comp_likelihoods[:, m] = scalar * np.exp(-0.5 * delta_2)
            
        numerator = comp_likelihoods * self.weights 
        
        denominator = numerator.sum(axis=1, keepdims=True) + self.TOL
        
        internal_gamma = numerator / denominator
        total_gamma = internal_gamma * state_posterior[:, np.newaxis]
        
        N_k = total_gamma.sum(axis=0) + self.TOL
        
        self.weights = N_k / (state_posterior.sum() + self.TOL)
        self.means = (total_gamma.T @ observations) / N_k[:, np.newaxis]
        
        for m in range(self.M):
            diff = observations - self.means[m] 
            weighted_diff = diff * total_gamma[:, m][:, np.newaxis]
            self.Sigmas[m] = (weighted_diff.T @ diff) / N_k[m]
            
            self.Sigmas[m] += np.eye(self.D) * 1e-3

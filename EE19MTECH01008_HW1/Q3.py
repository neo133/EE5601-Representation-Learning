"""
Q3) Implement EM algorithm for GMM parameter estimation
Author- Subhra Shankha Bhattacherjee
Roll - EE19MTECH01008
"""
import numpy as np

# function to generate dataset
def gen_data(d, N, K):
   
    init_mu = np.array([ np.random.uniform(high=k+1, low=k, size=d) for k in np.arange(K)])
    
    A = np.array([np.random.normal(1, 1, (d,d)) for k in np.arange(K)])
    init_sigma = np.array([np.matmul(P.T, P) for P in A])
    
    while(True): 
        k_dash = np.random.uniform(high=1, low=0, size=K)
        k_dash = N*k_dash/np.sum(k_dash)
        k_dash = np.rint(k_dash).astype(np.int)
        if np.sum(k_dash) == N : break
   
    mix = [np.random.multivariate_normal(init_mu[i], init_sigma[i], size=k_dash[i]) for i in np.arange(K)]
    
    X = np.concatenate(mix, axis=0)
    print("init mean:")
    print(init_mu)
    print("init cov:")
    print(init_sigma)
    return X, init_mu, init_sigma

#GMM class
class GMM():
    def __init__(self, X, k,tol):
        # dimension
        X = np.asarray(X)
        self.m, self.n = X.shape
        self.data = X.copy()
        self.k = k
        self.tol = tol
        
    def _init(self):
        # init mixture,means,sigmas
        self.mu_arr = np.asmatrix(np.random.random((self.k, self.n)))
        self.cov_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.pi = np.ones(self.k)/self.k
        self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
   
    def fit(self, tol):
        self._init()
        count = 0
        new_log_like = 1
        old_log_like = 0
        while(new_log_like-old_log_like) > tol:
            old_log_like = self.loglikelihood()
            self._fit()
            count += 1
            new_log_like = self.loglikelihood()
      
    def multivar_gauss_pdf(self, x, mu, sigma): # multivariate Gaussian pdf function
        n = len(x)
        p = 1/np.sqrt(((2*np.pi)**n)*np.linalg.det(sigma))
        p *= np.exp(-0.5*np.matmul(np.matmul((x-mu).T, np.linalg.inv(sigma)),(x-mu)))
        return p

    def loglikelihood(self):
        new_log_like = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
           
                tmp += self.pi[j]*self.multivar_gauss_pdf(self.data[i, :], 
                                                        self.mu_arr[j, :].A1, 
                                                        self.cov_arr[j, :])         
           
            log_sum = np.linalg.norm(tmp)
            new_log_like += log_sum 
        return new_log_like
  
    def _fit(self):
        self.e_step()
        self.m_step()
        
    #EM algo
    def e_step(self):
      
        for i in range(self.m):
            den = 0
            for j in range(self.k):
                num = self.multivar_gauss_pdf(self.data[i, :], 
                                                       self.mu_arr[j].A1, 
                                                       self.cov_arr[j]) *\
                      self.pi[j]
                den += num
                self.w[i, j] = num
            self.w[i, :] /= den
            assert self.w[i, :].sum() - 1 < tol
            
    def m_step(self):
        for j in range(self.k):
            const = self.w[:, j].sum()
            self.pi[j] = 1/self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.w[i, j])
                _sigma_j += self.w[i, j] * ((self.data[i, :] - self.mu_arr[j, :]).T * (self.data[i, :] - self.mu_arr[j, :]))
            self.mu_arr[j] = _mu_j / const
            self.cov_arr[j] = _sigma_j / const


print("enter dimension of gaussian d(>0):")
d=input()

print("enter clusters k(>0):")
K=input()

print("enter number of samples N(large value):")
N=input()

print("enter tolerance(very small value):")
tol=input()

X, means, covs = gen_data(d,N, K)

gmm = GMM(X,K,tol)
gmm.fit(tol)

print("estimated means")
print(gmm.mu_arr)
print("estimated sigmas")
print(gmm.cov_arr)
print("estimated weights")
print(gmm.pi)



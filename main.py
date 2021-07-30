import os 
import numpy as np 
import matplotlib.pyplot as plt 

# find the path
path = os.path.dirname(os.path.abspath(__file__))

class RW:

    def __init__( self, dim, params=[]):
        self.dim = dim 
        self._init_params(params)
        self._init_weights()
    
    def _init_params( self, params):
        if len(params):
            self.lr = params['lr']

    def _init_weights( self):
        self.W = np.zeros( [self.dim, 1])

    def step( self, xt, rt):
        '''
        1. predict reward:
            v =  ∑_d xt(1,d)w(d,1)
        2. calculate prediction error:
            δ = rt - v
        3. update params: 
            w += α * δ * xt.T
        '''
        v   = xt @ self.W              # 1d x d1 = 1
        rpe = rt - v                   # 1
        self.W += self.lr * rpe * xt.T # d1 += 1 x 1 x d1 

class KRW(RW):

    def __init__( self, dim, params=[]):
        super().__init__( dim)
        self._init_params(params)
    
    def _init_params(self, params):
        if len(params):
            self.cov_w = (params['sig_w']**2)* np.eye(self.dim)
            self.var_r = (params['sig_r']**2)
            self.cov_t = (params['tau']**2) * np.eye(self.dim)
    
    def step( self, xt, rt):
        '''
        1. predict reward:
            v =  ∑_d xt(1,d)w(d,1)
        2. calculate prediction error:
            δ = rt - v
        3. a priori cov
            cov_w += τ^2
        4. residual cov
            P = xt*cov_w*xt.T + cov_r
        5. Kalman gain 
            K = cov_w * xt.T / P
        6. update weight:
            w += K * δ
        7. update covariance
            cov_w -= K * xt * cov_w
        '''
        v   = xt @ self.W                       # 1d x d1 = 1
        rpe = rt - v 
        self.cov_w += self.cov_t                # dd + dd 
        P = xt @ self.cov_w @ xt.T + self.var_r # 1d x dd x d1 + 1 = 1
        K = self.cov_w @ xt.T / P               # dd x d1 / 1 = d1 
        self.W += K * rpe                       # d1 += d1 x 1
        self.cov_w -= K @ xt @ self.cov_w       # dd += d1 x 1d x dd 

class TD(RW):

    def __init__( self, dim, params=[]):
        super().__init__( dim)
        self._init_params(params)
    
    def _init_params( self, params):
        if len(params):
            self.lr = params['lr']
            self.g  = params['g']

    def _init_weights( self):
        self.W = np.zeros( [self.dim, 1])

    def step( self, xt, xt_next, rt):
        '''
        1. predict reward:
            vt =  ∑_d xt(1,d)w(d,1)
            vt+1 = ∑_d xt+1(1,d)w(d,1)
        2. calculate prediction error:
            δ = rt - vt + γ * vt+1
        3. update params: 
            w += α * δ * xt.T
        '''
        v = xt @ self.W                    # 1d x d1 = 1
        v_next = xt_next @ self.W          # 1d x d1 = 1
        rpe = rt - v + self.g * v_next     # 1 
        self.W += self.lr * rpe * (xt      # 1d += 1 x 1d
                - self.g * xt_next).T  

class KTD(RW):

    def __init__( self, dim, params=[]):
        super().__init__( dim)
        self._init_params(params)
    
    def _init_params(self, params):
        if len(params):
            self.cov_w = (params['sig_w']**2)* np.eye(self.dim)
            self.var_r = (params['sig_r']**2)
            self.cov_t = (params['tau']**2) * np.eye(self.dim)
            self.g     = params['gamma']
    
    def step( self, xt, xt_next, rt):
        '''
        1. predict reward:
            ht = xt - γ * xt+1 
            vt =  ∑_d xt(1,d)w(d,1)
            vt+1 = ∑_d xt+1(1,d)w(d,1)
        2. calculate prediction error:
            δ = rt - vt + γ * vt+1
        3. a priori cov
            cov_w += τ^2
        4. residual cov
            P = ht*cov_w*xt.T + cov_r
        5. Kalman gain 
            K = cov_w * ht.T / P
        6. update weight:
            w += K * δ
        7. update covariance
            cov_w -= K * ht * cov_w
        '''
        ht  = xt - self.g * xt_next             # 1d - 1d = 1d 
        rhat= ht @ self.W                       # 1d x d1 = 1
        rpe = rt - rhat                         # 1 - 1 = 1 
        self.cov_w += self.cov_t                # dd + dd 
        P = ht @ self.cov_w @ ht.T + self.var_r # 1d x dd x d1 + 1 = 1
        K = self.cov_w @ ht.T / P               # dd x d1 / 1 = d1 
        self.W += K * rpe                       # d1 += d1 x 1
        self.cov_w -= K @ ht @ self.cov_w       # dd += d1 x 1d x dd 

def simulation( effect):

    # construct stimuli 
    nT  = 10 
    dim = 2

    if effect == 'forward blocking':
        Xs  = np.ones( [ nT * 2, dim])
        Xs[ :nT, 1] = 0
        Rs  = np.ones( [ nT * 2, 1])
    elif effect == 'backward blocking':
        Xs  = np.ones( [ nT * 2, dim])
        Xs[ nT:nT*2, 1] = 0
        Rs  = np.ones( [ nT * 2, 1])
    else:
        raise KeyError

    # fix the parameters
    RWparams = dict()
    RWparams['lr'] = .3
    KRWparams = dict()
    KRWparams['sig_w'] = 1
    KRWparams['tau'] = .1
    KRWparams['sig_r'] = 1

    # init model 
    sub1 = RW( dim, RWparams)
    sub2 = KRW( dim, KRWparams)

    # storages for visualization
    W_RW  = np.zeros( [ 2*nT, dim]) + np.nan 
    W_KRW = np.zeros( [ 2*nT, dim]) + np.nan 

    # train the model
    for t in range(nT*2):

        # get observation and reward 
        xt = Xs[[t], :]
        rt = Rs[t]

        # train the model 
        sub1.step( xt, rt)
        sub2.step( xt, rt)

        # record the model learnt association
        W_RW[ t, :]  = sub1.W.reshape([-1])
        W_KRW[ t, :] = sub2.W.reshape([-1])

    # viualization 
    plt.figure()
    plt.plot( W_RW, linewidth = 3)
    plt.plot( W_KRW, '--', linewidth = 3)
    plt.xlabel( 'Trial', fontsize = 25)
    plt.ylabel( 'Weight', fontsize = 25)
    plt.legend( [ 'A (RW)', 'B (RW)','A (KRW)','B (KRW)'])
    plt.savefig( f'{path}/{effect}.png', dpi = 1000)

if __name__ == '__main__':

    simulation('forward blocking')
    simulation('backward blocking')
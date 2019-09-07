import time, math, torch, shutil, glob
import numpy as np
import os
import random, string, os
import glob, copy
import Optimizer.lib.optimizer as optimizers
from scipy.stats import ortho_group

def crossed_zero(old, new):
    return (old * new <= 0).float()

def random_string(N=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(N))

def to_cuda(x):
    try:
        iterator = iter(x)
    except TypeError:
        # not iterable
        x = x.cuda()
    else:
        # iterable
        x = [x_.cuda() for x_ in x]
    return x

class CrossZeroTracker(object):
    def __init__(self, optimizer):
        self.opt = optimizer
        self.beta1 = 0.9
        self.n_grad_flip = 0 # number of momentum flip
        self.momentum = {}
        for group in self.opt.param_groups:
            for p in group['params']:
                self.momentum[p] = torch.zeros_like(p)
                
    def record(self):
        for group in self.opt.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                exp_avg = self.momentum[p]
                self.n_grad_flip += torch.sum(crossed_zero(exp_avg,
                                                           exp_avg*self.beta1\
                                                           + (1-self.beta1)*grad)).item()
                exp_avg.mul_(self.beta1).add_(1 - self.beta1, grad)
        
class OptPath():
    def __init__(self, max_iter=1000, tol=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        
    def get_path(self, criteria, x0, lr=1e-3, opt=torch.optim.SGD, sgd_adjust=False,
                 schedule=None, decay_rate=0.1, crosszero=False, record=False, **kwargs):
        self.lr = lr
        self.kwargs = kwargs
        self.schedule = schedule
        self.decay_rate = decay_rate
        self.crosszero = crosszero
        self.criteria = criteria
        self.validation_decay = schedule is not None and 0 in schedule
        
        x_path = [x0]
        x = torch.nn.Parameter(torch.from_numpy(x0).float().view(1, -1))
        optimizer = opt([x], lr=lr, **kwargs)
        self.opt = optimizer
        self.opt_recorder = OptRecorder(optimizer)
        if sgd_adjust: self.sgd_adjuster = optimizers.SGD_adjust(self.opt)
        
        if schedule is not None:
            if self.validation_decay:
                self.lr_decay = torch.optim.lr_scheduler.\
                                ReduceLROnPlateau(optimizer, mode='min',
                                                  factor=decay_rate,
                                                  patience=10, cooldown=100)
            else:
                self.lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                     schedule,
                                                                     gamma=decay_rate)
            
        for i in range(self.max_iter):
            optimizer.zero_grad()
            l = criteria(x)
            l.backward()
            if sgd_adjust: self.sgd_adjuster.step(l.data.item())
            optimizer.step()

            x_path.append(x.data.cpu().clone().numpy().ravel())
            if schedule is not None:
                if self.validation_decay:
                    self.lr_decay.step(l.data.item())
                else:
                    self.lr_decay.step()
                    
            if record:
                self.opt_recorder.record()
        
        x_path = np.vstack(x_path)
        self.x_path = x_path
        
    def get_converge_time(self, tol=None):
        if tol is None: tol = self.tol
        # definition of converge here is after certain iterations,
        # the distance to 0 is smaller than tol
        t = len(self.x_path)
        for i, pt in enumerate(self.x_path[::-1]):
            l = np.linalg.norm(pt)
            if np.isnan(l):
                return len(self.x_path)
            if l > tol:
                return len(self.x_path) - i
        return 0
    
    def get_loss_time(self, tol=None):
        if tol is None: tol = self.tol
        t = len(self.x_path)
        for i, pt in enumerate(self.x_path[::-1]):
            l = self.criteria(torch.from_numpy(pt).view(1, -1).float()).item()
            if np.isnan(l):
                return len(self.x_path) * 2
            if l > tol:
                return len(self.x_path) - i
        return 0

    def get_loss(self):
        return [self.criteria(torch.from_numpy(x).view(1, -1).float()).item() for
                x in self.x_path]

def gen_quadratic_data(n, d=2, lambda_min=1, lambda_max=1, logscale=False,
                       theta_star=None, Q=None, offset=0):
    '''
    see distill.ipynb
    '''
    assert n > d, "n > d otherwise ill condition for this illustrative plot"

    if Q is None:
        Q = ortho_group.rvs(d)
        
    U = ortho_group.rvs(n)
    if not logscale:
        Lambda = np.linspace(lambda_min, lambda_max, d)
    else:
        Lambda = np.logspace(np.log10(lambda_min), np.log10(lambda_max), d)
        
    Sigma = np.sqrt(Lambda)
    X = U[:, :d].dot(np.diag(Sigma)).dot(Q)
    if theta_star is None:
        theta_star = np.random.randint(5, size=(d, 1))
    #y = np.random.randn(n).reshape(-1,1)
    y = X.dot(theta_star) + offset
    return Q, Lambda, X, y

def gen_criteria(X, y):
    D = torch.from_numpy(X).float()
    target = torch.from_numpy(y).float()
    def res(x): # x is  axd
        error = D.mm(x.transpose(0,1)) - target # n by a
        ret = 0.5 * (error * error).sum(0) / len(X)
        return ret
    
    theta_star = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return res, theta_star
                                                    
def gen_quadratic_loss(d=10, lambda_min=1, lambda_max=1, logscale=False, Q=None):
    '''generate quadratic loss for synthetic data'''
    if Q is None:
        Q = ortho_group.rvs(d)
    if not logscale:
        Lambda = np.linspace(lambda_min, lambda_max, d)
    else:
        Lambda = np.logspace(np.log10(lambda_min), np.log10(lambda_max), d)
    A = torch.from_numpy(Q.T.dot(np.diag(Lambda)).dot(Q)).float()
    # x is n by d
    def res(x):
        return 0.5 * (x.mm(A) * x).sum(1)
    
    return res, Q, Lambda

class OptRecorder(object):
    """collect items in optimizer"""
    def __init__(self, optimizer, n=10, model=None):
        if model is not None:
            self.w0 = copy.deepcopy(model.state_dict())
            
        self.opt = optimizer
        self.n = n # number of tracked parameters
        # randomly choose n parameters for each layer
        self.index = {}
        self.tracker = []

        for group in optimizer.param_groups:
            for p in group['params']:
                length = len(p.data.cpu().detach().numpy().ravel())
                ntrack = min(n, length)
                if n >= length:
                    self.index[p] = list(range(length))
                else:
                    self.index[p] = np.random.choice(range(length), ntrack, replace=False)
                self.tracker.append({
                    "grad": [[] for _ in range(ntrack)],
                    "param": [[] for _ in range(ntrack)],
                    "alpha_ratio": [[] for _ in range(ntrack)],
                    "feature_step": [[] for _ in range(ntrack)],
                    "lr": [[] for _ in range(ntrack)],         
                })

        # last item of tracker keeps track of global properties
        if model is not None:
            self.tracker.append({"l2(w-w0)": [], "l2(w)": []})

    def record(self, model=None):
        ind = 0
        if model is not None:
            self.tracker[-1]["l2(w-w0)"].append(0)
            self.tracker[-1]["l2(w)"].append(0)
            w = model.state_dict()
            for k in w.keys():
                self.tracker[-1]["l2(w-w0)"][-1]+=torch.sum((w[k]-self.w0[k])**2).item()
                self.tracker[-1]["l2(w)"][-1] += torch.sum(w[k]**2).item()

            self.tracker[-1]["l2(w-w0)"][-1] = np.sqrt(self.tracker[-1]["l2(w-w0)"][-1])
            self.tracker[-1]["l2(w)"][-1] = np.sqrt(self.tracker[-1]["l2(w)"][-1])

        for group in self.opt.param_groups:
            for param in group['params']:
                state = self.opt.state[param]

                p = param.data.cpu().detach().numpy().ravel()

                if param.grad is not None:
                    g = param.grad.data.cpu().detach().numpy().ravel()
                else:
                    g = np.ones_like(p)
                    
                if 'alpha_ratio' in state:
                    a = state['alpha_ratio'].cpu().detach().numpy().ravel()
                else:
                    a = np.ones_like(p)

                if 'feature_step' in state:
                    f = state['feature_step'].cpu().detach().numpy().ravel()
                elif 'step' in state:
                    f = np.ones_like(p) * state['step']
                else:
                    f = np.ones_like(p)

                if 'lr' in state:
                    lr = state['lr'].cpu().detach().numpy().ravel()
                else:
                    lr = np.ones_like(p) * group['lr']
                    
                for i, index in enumerate(self.index[param]):
                    self.tracker[ind]['grad'][i].append(g[index])
                    self.tracker[ind]['param'][i].append(p[index])                    
                    self.tracker[ind]['alpha_ratio'][i].append(a[index])
                    self.tracker[ind]['feature_step'][i].append(f[index])
                    self.tracker[ind]['lr'][i].append(lr[index])   
                ind += 1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum square
        self.var = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sum_2 += val**2 * n
        self.var = self.sum_2 / self.count - self.avg**2

class PrintTable(object):
    '''print tabular data in a nice format'''
    def __init__(self, nstr=15, nfloat=5):
        self.nfloat = nfloat
        self.nstr = nstr

    def _format(self, x):
        if type(x) is float:
            x_tr = ("%." + str(self.nfloat) + "f") % x
        else:
            x_tr = str(x)
        return ("%" + str(self.nstr) + "s") % x_tr

    def print(self, row):
        print( "|".join([self._format(x) for x in row]) )

def smooth(sequence, step=1):
    out = np.convolve(sequence, np.ones(step), 'valid') / step
    return out

def random_split_dataset(dataset, proportions, seed=None):
    n = len(dataset)
    ns = [int(math.floor(p*n)) for p in proportions]
    ns[-1] += n - sum(ns)

    def random_split(dataset, lengths):
        if sum(lengths) != len(dataset):
            raise ValueError("Sum of input lengths does not equal\
            the length of the input dataset!")

        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(sum(lengths))
        return [torch.utils.data.Subset(dataset, indices[offset - length:offset])\
                for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
    #return torch.utils.data.random_split(dataset, ns)
    return random_split(dataset, ns)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


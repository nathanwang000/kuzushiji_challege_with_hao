import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import math, copy
from collections import defaultdict

def crossed_zero(old, new):
    return (old * new <= 0).float()

def reset_grad(optimizer): # not zero_grad as in optimizer default
    # refresh gradient at each epoch
    for group in optimizer.param_groups:        
        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            state['grad'] = torch.zeros_like(p.data)

class AdaSGD(Optimizer): # per dimension SGD
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, beta=0.999, eps=1e-9, amsgrad=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        beta=beta, eps=eps, amsgrad=amsgrad)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdaSGD, self).__init__(params, defaults)

        self.w_sq = 0 # weight squared divided by d
        self.max_w_sq = 0
        self.beta = beta
        self.step_ = 0        

    def __setstate__(self, state):
        super(AdaSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('amsgrad', False)
            
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # calculate norm of the vector
        w_sq = 0
        d = 0
        self.step_ += 1
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0:
                    grad += weight_decay * p.data
                w_sq += (grad**2).sum()
                d += grad.numel()
                
        self.w_sq = self.w_sq * self.beta + (1-self.beta) * (w_sq / d)

        # actual update
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                bias_correction = 1 - self.beta ** self.step_
                if amsgrad:
                    if type(self.max_w_sq) is int:
                        self.max_w_sq = copy.deepcopy(self.w_sq)
                    else:
                        self.max_w_sq = torch.max(self.max_w_sq, self.w_sq)
                    denom = self.max_w_sq.sqrt() + group['eps']
                else:
                    denom = self.w_sq.sqrt() + group['eps']

                denom /= math.sqrt(bias_correction)
                p.data.addcdiv_(-group['lr'], d_p, denom)
                #p.data.add_(-group['lr'], d_p) # old                

        return loss
            
class dSGD(Optimizer): # per dimension SGD
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(dSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(dSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, update=True):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['lr'] = torch.ones_like(p.data) * group["lr"]
                    state['grad'] = torch.zeros_like(p.data)

                state['grad'] += p.grad.data                    
                if not update:
                    continue
                
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # p.data.add_(-group['lr'], d_p)
                p.data.add_(-state['lr'] * d_p)

        return loss

class AlphaSGD(Optimizer):
    '''
    add Alpha term to SGD with variance normalization
    '''
    def __init__(self, params, lr=1e-3, alphas=(1, 1), betas=(0.9, 0.999),
                 eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, alphas=alphas, eps=eps, max_lr=max_lr)
        super(AlphaSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(g2)'] = torch.zeros_like(p.data)
                    state['EMA(w2)'] = torch.zeros_like(p.data)                    
                    state['var(w)'] = torch.zeros_like(p.data)                    
                    state['var(g)'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                alpha1, alpha2 = group['alphas']                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # calculate variance
                dg = grad - state['EMA(g2)']
                state['var(g)'] = beta2 * (state['var(g)'] + (1-beta2) * dg**2)
                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)
                state['EMA(g2)'].mul_(beta2).add_(1-beta2, grad)

                dw = p.data - state['EMA(w2)']
                state['var(w)'] = beta2 * (state['var(w)'] + (1-beta2) * dw**2)
                state['EMA(w2)'].mul_(beta2).add_(1-beta2, p.data)      

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_g2 = state['EMA(g2)'] / bias_correction2
                var_g = state['var(g)'] / bias_correction2
                var_w = state['var(w)'] / bias_correction2

                curvature = var_g
                denom = (alpha1 + alpha2 * curvature).sqrt().add_(group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = torch.sqrt(alpha1 / (alpha2 * curvature))
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                

        return loss

class AlphaAdam(Optimizer):
    '''
    add Alpha term to adam
    '''
    def __init__(self, params, lr=1e-3, alphas=(1, 1), betas=(0.9, 0.999),
                 eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, alphas=alphas, eps=eps, max_lr=max_lr)
        super(AlphaAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(g2)'] = torch.zeros_like(p.data)
                    state['var(g)'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                alpha1, alpha2 = group['alphas']                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # calculate variance
                dg = grad - state['EMA(g2)']
                state['var(g)'] = beta2 * (state['var(g)'] + (1-beta2) * dg**2)
                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)
                state['EMA(g2)'].mul_(beta2).add_(1-beta2, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_g2 = state['EMA(g2)'] / math.sqrt(bias_correction2)
                var_g = state['var(g)'] / bias_correction2

                denom = (alpha1 * E_g2**2 + alpha2 * var_g).sqrt().add_(group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = torch.sqrt(alpha1*E_g2**2/(alpha2*var_g))
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                

        return loss

class AlphaDiff(Optimizer):
    '''
    add Alpha term to diff
    '''
    def __init__(self, params, lr=1e-3, alphas=(1, 1), betas=(0.9, 0.999),
                 eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, alphas=alphas, eps=eps, max_lr=max_lr)
        super(AlphaDiff, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(dg)'] = torch.zeros_like(p.data)
                    state['var(dg)'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                alpha1, alpha2 = group['alphas']                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # calculate variance
                dg = grad - state['last_grad']
                ddg = dg - state['EMA(dg)']
                state['var(dg)'] = beta2 * (state['var(dg)'] + (1-beta2) * ddg**2)

                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)                
                state['EMA(dg)'].mul_(beta2).add_(1-beta2, dg)                

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_dg = state['EMA(dg)'] / math.sqrt(bias_correction2)
                var_dg = state['var(dg)'] / bias_correction2

                denom = (alpha1 * var_dg + alpha2 * E_dg**2).sqrt().add_(group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = torch.sqrt(alpha1*var_dg/(alpha2*E_dg**2))
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                

        return loss

class AdamC1(Optimizer):
    '''
    var(g) / var(w)
    '''
    def __init__(self, params, lr=1e-3, alphas=(1, 1), betas=(0.9, 0.999),
                 eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, alphas=alphas, eps=eps, max_lr=max_lr)
        super(AdamC1, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(g2)'] = torch.zeros_like(p.data)
                    state['EMA(w2)'] = torch.zeros_like(p.data)                    
                    state['var(w)'] = torch.zeros_like(p.data)                    
                    state['var(g)'] = torch.zeros_like(p.data)
                    state['EMA(dg)'] = torch.zeros_like(p.data)
                    state['var(dg)'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                alpha1, alpha2 = group['alphas']                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                dg = grad - state['last_grad']
                ddg = dg - state['EMA(dg)']
                state['var(dg)'] = beta2 * (state['var(dg)'] + (1-beta2) * ddg**2)
                state['var(g)'] = beta2 * (state['var(g)'] +
                                           (1-beta2) * (grad-state['EMA(g2)'])**2) 

                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)
                state['EMA(g2)'].mul_(beta2).add_(1-beta2, grad) 
                state['EMA(dg)'].mul_(beta2).add_(1-beta2, dg)
                
                dw = p.data - state['EMA(w2)']
                state['var(w)'] = beta2 * (state['var(w)'] + (1-beta2) * dw**2)
                state['EMA(w2)'].mul_(beta2).add_(1-beta2, p.data)      

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_g2 = state['EMA(g2)'] / math.sqrt(bias_correction2)
                E_dg = state['EMA(dg)'] / math.sqrt(bias_correction2)                
                var_g = state['var(g)'] / bias_correction2
                var_w = state['var(w)'] / bias_correction2

                curvature = var_g / var_w
                denom = (alpha1*E_g2**2 + alpha2 * curvature).sqrt().add_(group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = torch.sqrt(alpha1*E_g2**2 / (alpha2 * curvature))
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                

        return loss
    
class AdamC2(Optimizer):
    '''
    E(dg)**2
    '''
    def __init__(self, params, lr=1e-3, alphas=(1, 1), betas=(0.9, 0.999),
                 eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, alphas=alphas, eps=eps, max_lr=max_lr)
        super(AdamC2, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(g2)'] = torch.zeros_like(p.data)
                    state['EMA(w2)'] = torch.zeros_like(p.data)                    
                    state['var(w)'] = torch.zeros_like(p.data)                    
                    state['var(g)'] = torch.zeros_like(p.data)
                    state['EMA(dg)'] = torch.zeros_like(p.data)
                    state['var(dg)'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                alpha1, alpha2 = group['alphas']                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                dg = grad - state['last_grad']
                ddg = dg - state['EMA(dg)']
                state['var(dg)'] = beta2 * (state['var(dg)'] + (1-beta2) * ddg**2)
                state['var(g)'] = beta2 * (state['var(g)'] +
                                           (1-beta2) * (grad-state['EMA(g2)'])**2) 

                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)
                state['EMA(g2)'].mul_(beta2).add_(1-beta2, grad) 
                state['EMA(dg)'].mul_(beta2).add_(1-beta2, dg)
                
                dw = p.data - state['EMA(w2)']
                state['var(w)'] = beta2 * (state['var(w)'] + (1-beta2) * dw**2)
                state['EMA(w2)'].mul_(beta2).add_(1-beta2, p.data)      

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_g2 = state['EMA(g2)'] / math.sqrt(bias_correction2)
                E_dg = state['EMA(dg)'] / math.sqrt(bias_correction2)                
                var_g = state['var(g)'] / bias_correction2
                var_w = state['var(w)'] / bias_correction2

                curvature = E_dg**2 
                denom = (alpha1*E_g2**2 + alpha2 * curvature).sqrt().add_(group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = torch.sqrt(alpha1*E_g2**2 / (alpha2 * curvature))
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                

        return loss
    
class Sign(Optimizer):
    '''
    only use the sign of gradient
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Sign, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                bias_correction1 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction1

                p.data.add_(-step_size, torch.sign(exp_avg))

        return loss

class NormalizedCurvature2(Optimizer):
    '''
    (gt - g{t-1} / m_t)^2
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(NormalizedCurvature2, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                mhat = exp_avg / bias_correction1
                diff = (grad - state['last_grad'] / mhat)**2
                
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                p.data.add_(-numer * exp_avg)

        return loss
    
class NormalizedCurvature(Optimizer):
    '''
    (gt - g{t-1} / (w_t - w_{t-1}))^2
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(NormalizedCurvature, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)
                    state['w_diff'] = torch.ones_like(p.data)
                    state['grad_diff'] = torch.ones_like(p.data) 

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    w_diff = torch.ones_like(p.data)  # don't use diff at first step
                else:
                    w_diff = torch.abs(p.data - state['last_w']) + group['eps']  

                diff = ((grad - state['last_grad']) / w_diff)**2
                # state['w_diff'].mul_(beta2).add_(1-beta2, w_diff)
                # grad_diff = torch.abs(grad - state['last_grad'])
                # state['grad_diff'].mul_(beta2).add_(1-beta2, grad_diff)
                # pct_grad_diff = grad_diff / state['grad_diff']
                # diff = pct_grad_diff**2
                # pct_wdiff = w_diff / state['w_diff']
                #diff = (pct_grad_diff / pct_wdiff)**2
                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # step_size = 0
                numer = torch.max(step_size / denom, 1e-8 * torch.ones_like(exp_avg))
                #numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()

                p.data.add_(-numer * exp_avg)

        return loss

class CurvatureSign(Optimizer):
    '''
    lr * / |(g_t - g_{t-1})| * sign(m_t)
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(CurvatureSign, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                #denom = torch.abs(p.data - state['last_w']) + group['eps']
                #diff = ((grad - state['last_grad']) / denom)**2
                diff = (grad - state['last_grad'])**2                
                # print(torch.min(p.data - state['last_w']))
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                
                p.data.add_(-numer * torch.sign(exp_avg))

        return loss

class RCSign(Optimizer):
    '''
    Sign method normalized by relative cuvature (fix issue with gradient)
    lr * |min(g_t, g_{t-1}) / (g_t - g_{t-1})| * sign(m_t)
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(RCSign, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                denom = torch.abs(grad)#(torch.abs(grad) + torch.abs(state['last_grad'])) / 2
                #denom = torch.min(torch.abs(grad), torch.abs(state['last_grad']))                
                #diff = ((grad - state['last_grad']) / denom)**2
                diff = torch.abs((grad - state['last_grad']) / denom)
                #diff = torch.min(diff, 10**6 * torch.ones_like(exp_avg))

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                #denom = exp_avg_sq.sqrt().add_(group['eps'])
                denom = exp_avg_sq.add_(group['eps'])                 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                #step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                step_size = group['lr'] * bias_correction2 / bias_correction1          

                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                
                p.data.add_(-numer * torch.sign(exp_avg))

        return loss

class EffectiveMCSign(Optimizer):
    '''
    (gt - g{t-1})^2
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, threshold=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, threshold=threshold)
        super(EffectiveMCSign, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['effective_lr'] = torch.ones_like(p.data) * group['lr']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - state['last_grad'])**2                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['threshold'] * math.sqrt(bias_correction2) / bias_correction1
                ratio = step_size / denom

                ratio[ratio > 1] = 1.1
                ratio[ratio < 1] = 0.99
                state['effective_lr'] = torch.min(torch.max(state['effective_lr'] * ratio,
                                                            group['lr'] * torch.ones_like(p.data)),
                                                  100 * group['lr'] * torch.ones_like(p.data)) # at most increase by 10
                
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                p.data.add_(-state['effective_lr'] * exp_avg)

        return loss

class SecondMoment(Optimizer):
    '''
    no variance, just divide by abs(EMA(gt-gt-1))
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(SecondMoment, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * bias_correction2 / bias_correction1

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                diff = (grad - state['last_grad']) / (exp_avg / bias_correction1)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = torch.abs(exp_avg_sq).add_(group['eps'])
                #denom = torch.abs(exp_avg_sq).add_(1) 

                numer = step_size / denom
                #numer = torch.min(step_size / denom,
                #                  1e-3 * torch.ones_like(exp_avg))

                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                p.data.add_(-numer * exp_avg)

        return loss

class DoubleMomentum(Optimizer):
    '''
    numerator contains another momentum
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(DoubleMomentum, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                #denom = torch.abs(p.data - state['last_w']) + group['eps']
                #diff = ((grad - state['last_grad']) / denom)**2
                diff = (grad - state['last_grad'])**2                
                # print(torch.min(p.data - state['last_w']))
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                
                p.data.add_(-numer * exp_avg * torch.abs(exp_avg))

        return loss

class AlphaDiff2(Optimizer):
    '''
    add Alpha term to diff, and also use var to measure 2nd order info
    '''
    def __init__(self, params, lr=1e-3, alphas=(1, 1), betas=(0.9, 0.999),
                 eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, alphas=alphas, eps=eps, max_lr=max_lr)
        super(AlphaDiff2, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(dg)'] = torch.zeros_like(p.data)
                    state['var(dg)'] = torch.zeros_like(p.data)
                    state['EMA(g2)'] = torch.zeros_like(p.data)
                    state['var(g)'] = torch.zeros_like(p.data)
                    
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                alpha1, alpha2 = group['alphas']                
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # calculate variance
                dg = grad - state['last_grad']
                ddg = dg - state['EMA(dg)']
                state['var(dg)'] = beta2 * (state['var(dg)'] + (1-beta2) * ddg**2)
                state['var(g)'] = beta2 * (state['var(g)'] +
                                           (1-beta2) * (grad-state['EMA(g2)'])**2) 

                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)
                state['EMA(g2)'].mul_(beta2).add_(1-beta2, grad) 
                state['EMA(dg)'].mul_(beta2).add_(1-beta2, dg)


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_dg = state['EMA(dg)'] / bias_correction2
                E_g2 = state['EMA(g2)'] / bias_correction2                
                var_g = state['var(g)'] / bias_correction2
                var_dg = state['var(dg)'] / bias_correction2

                denom = (alpha1 * var_dg + alpha2 * var_g).sqrt().add_(group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = torch.sqrt(alpha1*var_dg/(alpha2*var_g))
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                

        return loss

class Diff(Optimizer):
    '''
    use V to record difference between estimates instead!
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Diff, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['last_grad'] = torch.zeros_like(p.data)                    

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - state['last_grad'])**2
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()

        return loss
    
class MomentumCurvature2(Optimizer):
    '''
    abs(gt - g{t-1})
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(MomentumCurvature2, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                diff = torch.abs(grad - state['last_grad'])
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.add_(group['eps'])  # no square

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * bias_correction2 / bias_correction1

                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                p.data.add_(-numer * exp_avg)

        return loss

class MomentumCurvature3(Optimizer):
    '''
    (E(gt - g{t-1}))**2
    this effectively just makes the learning rate of MC times 10 b/c 1 - betas[0] = 0.1
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(MomentumCurvature3, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_diff'] = torch.zeros_like(p.data)                    
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']
                beta1, beta2, beta3 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
     
                diff = grad - state['last_grad']
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_diff.mul_(beta1).add_(1 - beta3, diff)
                # exp_avg_sq.mul_(beta2).add_(1 - beta2, exp_avg_diff**2)
                exp_avg_sq.mul_(beta2).add_(1 - beta2,
                                            (exp_avg_diff / bias_correction3)**2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                numer = step_size / denom * torch.abs(exp_avg)
                if group['max_lr']:
                    numer = torch.min(numer,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                    
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                p.data.add_(-numer * torch.sign(exp_avg))

        return loss

class Adam3(Optimizer):
    '''
    E(gt)**2
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(Adam3, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
     
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).add_(1 - beta2, exp_avg**2)
                exp_avg_sq.mul_(beta2).add_(1 - beta2,
                                            (exp_avg / bias_correction1)**2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                numer = step_size / denom * torch.abs(exp_avg)
                if group['max_lr']:
                    numer = torch.min(numer,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                    
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-numer * torch.sign(exp_avg))

        return loss
    
class RK4(Optimizer):
    
    def __init__(self, params, lr=required, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(RK4, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK4, self).__setstate__(state)

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        
        for i in range(4):
            loss = closure() # recompute gradient
                
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is None:
                        continue
                
                    # save k1 to k4 to param_state
                    param_state = self.state[p]
                    if 'k' not in param_state:
                        param_state['k'] = [0] * 4 # list of size 4
                    param_state['k'][i] = p.grad.data
                    
                    coefs = [1, 0.5, 0.5, 1]
                    # undo last update
                    if i != 0: # add back gradient
                        p.data.add_(group['lr'] * coefs[i], param_state['k'][i-1])
                    # update
                    if i != 3: # intermediate update
                        p.data.add_(-group['lr'] * coefs[i+1], param_state['k'][i])
                    else: # real update
                        k1, k2, k3, k4 = param_state['k']
                        p.data.add_(-group['lr'] / 6, k1 + 2*k2 + 2*k3 + k4) 
        
        return loss
    

class DoublingRK4(Optimizer):
    
    '''
    take 1 full step and 2 half step and compare the difference,
    not a proper implementation of Doubling RK4 as 
    '''
    def __init__(self, params, lr=required, weight_decay=0, tol=1e-7):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, tol=tol)
        super(DoublingRK4, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK4, self).__setstate__(state)

    def step_scale(self, closure, scale):
        loss = None
        
        for i in range(4):
            loss = closure() # recompute gradient
                
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is None:
                        continue
                
                    # save k1 to k4 to param_state
                    param_state = self.state[p]
                    if 'k' not in param_state:
                        param_state['k'] = [0] * 4 # list of size 4
                    param_state['k'][i] = p.grad.data
                    
                    coefs = [1, 0.5, 0.5, 1]
                    # undo last update
                    if i != 0: # add back gradient
                        p.data.add_(group['lr'] * coefs[i] * scale, param_state['k'][i-1])
                    # update
                    if i != 3: # intermediate update
                        p.data.add_(-group['lr'] * coefs[i+1] * scale, param_state['k'][i])
                    else: # real update
                        k1, k2, k3, k4 = param_state['k']
                        p.data.add_(-group['lr'] / 6 * scale, k1 + 2*k2 + 2*k3 + k4)

                        # save this update: to be compared later
                        savename = 'update{}'.format(scale)
                        if savename not in param_state:
                            param_state[savename] = []
                        param_state[savename].append(-group['lr'] / 6 * scale * (k1 + 2*k2 + 2*k3 + k4))
                        
        return loss

    def undo_step(self, nsteps, scale):
        '''
        nsteps is how many steps back to undo
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                param_state = self.state[p]
                savename = 'update{}'.format(scale)
                for i, change in enumerate(param_state[savename][::-1]):
                    if i > nsteps:
                        break
                    p.data.add_(-change)
        
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # take 1 large step
        self.step_scale(closure, 2) 
        # undo the step
        self.undo_step(nsteps=1, scale=2)
        
        # take 2 small steps
        self.step_scale(closure, 1)
        loss = self.step_scale(closure, 1)
        
        # compare steps
        for group in self.param_groups:
            tol = group['tol'] # tolerence get from defaults

            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                
                y2 = sum(param_state['update2'])
                y1 = sum(param_state['update1'])
                #print(y1.shape, y2.shape)
                
                # find max deviation
                delta = torch.max(torch.abs(y2 - y1))

                factor = (tol / delta)**(1/5)
                tmplr = group['lr'] * factor
                group['lr'] = min(max(tmplr, 1e-8), 0.1) # this seems inefficient
                #print(group['lr'], delta)
                #print(group['lr'])
                
                # restore update to none
                param_state['update1'] = []
                param_state['update2'] = []

        return loss

class Avrng(Optimizer):
    '''
    adaptive variance reduced & (curvature) normalized gradient
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(Avrng, self).__init__(params, defaults)

    def rewind(self):
        '''
        rewind 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # last change
                    state['delta'] = torch.zeros_like(p.data)

                p.data.add_(-state['delta']) # reverse change

    def forward(self):
        '''
        forward 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                p.data.add_(state['delta']) # forward change

                # save old grad
                state['old_grad'] = p.grad.data.clone()
                
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # last time step
        self.rewind()
        closure()
        # go back to now and save grad to old_grad
        self.forward()
        loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
               
                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    old_grad = torch.zeros_like(state['old_grad'])
                else:
                    old_grad = state['old_grad']
                
                # variance reduced gradient:
                diff = (grad - old_grad)**2 # capture curvature, not variance
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                delta = -numer * exp_avg
                state['delta'] = delta
                p.data.add_(delta)
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return [loss, loss] # two times

class AdamVR(Optimizer):
    '''
    adaptive variance reduced & (curvature) normalized gradient
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(AdamVR, self).__init__(params, defaults)

    def rewind(self):
        '''
        rewind 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # last change
                    state['delta'] = torch.zeros_like(p.data)

                p.data.add_(-state['delta']) # reverse change

    def forward(self):
        '''
        forward 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                p.data.add_(state['delta']) # forward change

                # save old grad
                state['old_grad'] = p.grad.data.clone()
                
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # last time step
        self.rewind()
        closure()
        # go back to now and save grad to old_grad
        self.forward()
        loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
               
                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                # variance reduced gradient:
                gradhat = grad - state['old_grad'] + mhat
                    
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, gradhat**2)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                numer = torch.min(step_size / denom, group['max_lr'] * torch.ones_like(exp_avg))
                delta = -numer * exp_avg
                state['delta'] = delta
                p.data.add_(delta)
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return [loss, loss] # two times
    
class Adam2(Optimizer):
    '''
    do 2 Adam update
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=1):
        self.adam = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)

    def zero_grad(self):
        self.adam.zero_grad()
        
    def step(self, closure):
        loss1 = self.adam.step(closure)
        loss2 = self.adam.step(closure)
        return [loss1, loss2]
        
class Avrng2(Optimizer):
    '''
    adaptive variance reduced & (curvature) normalized gradient
    do twice, without wasting any update
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(Avrng2, self).__init__(params, defaults)

    def update1(self):
        '''
        rewind 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                diff = (grad - mhat)**2 # capture curvature, not variance
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                #numer = torch.min(step_size / denom,
                #                  group['max_lr'] * torch.ones_like(exp_avg))
                #p.data.add_(-numer * exp_avg)
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
                # save old grad
                state['old_grad'] = p.grad.data.clone()
                
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss1 = closure()
        self.update1()

        loss2 = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
               
                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                # variance reduced gradient:
                gradhat = grad - state['old_grad'] + mhat
                    
                diff = (gradhat - mhat)**2 # capture curvature, not variance
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                # numer = torch.min(step_size / denom,
                #                   group['max_lr'] * torch.ones_like(exp_avg))
                # p.data.add_(-numer * exp_avg)
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return [loss1, loss2]
    
class Diff2(Optimizer):
    '''
    use V to record difference between estimates instead!
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Diff2, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                diff = (grad - mhat)**2 # this is the only change from Adam!!!!
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class DiffUnbiased(Optimizer):
    '''
    use V to record difference between estimates instead!
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(DiffUnbiased, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # note: use actual value
                    state['exp_avg'] = grad.clone()
                    # Exponential moving average of squared difference in gradient values
                    # note: use one step look ahead: that is not update for the first
                    # step
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - exp_avg)**2 # this is the only change from Adam!!!!   
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                if state['step'] == 2:
                    state['exp_avg_sq'] = diff.clone()
                    exp_avg_sq = state['exp_avg_sq']
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                    
                # note: denom is per term, doesnt sound
                if state['step'] <= 1: # accumulate for a few steps
                    denom = torch.ones_like(p.data)
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(-group['lr'], exp_avg, denom)

        return loss

class DiffUnbiasedBounded(Optimizer):
    '''
    use V to record difference between estimates instead!
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, upper_bound=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, upper_bound=upper_bound)
        super(DiffUnbiasedBounded, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # note: use actual value
                    state['exp_avg'] = grad.clone()
                    # Exponential moving average of squared difference in gradient values
                    # note: use one step look ahead: that is not update for the first
                    # step
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - exp_avg)**2 # this is the only change from Adam!!!!   
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                if state['step'] == 2:
                    state['exp_avg_sq'] = diff.clone()
                    exp_avg_sq = state['exp_avg_sq']
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                    
                # note: denom is per term, doesnt sound
                if state['step'] <= 1: # accumulate for a few steps
                    denom = torch.ones_like(p.data)
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                numer = torch.min(group['lr'] / denom,
                                  group['upper_bound'] * torch.ones_like(exp_avg))
                p.data.add_(-exp_avg * numer)

        return loss
    
class DiffMax(Optimizer):
    '''
    changed per parameter update to all together update
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(DiffMax, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - exp_avg)**2 # this is the only change from Adam!!!! 
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                # only difference from Diff1 is to make denom a scaler
                denom = torch.max(exp_avg_sq.sqrt()).add_(group['eps']) # only difference from Diff

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class AdamUnbiased(Optimizer):
    
    '''
    changed unbiasing operation from Adam
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamUnbiased, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamUnbiased, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad #torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad**2#torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                #bias_correction1 = 1 - beta1 ** state['step']
                #bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr']# * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                state['alpha_ratio'] = torch.ones_like(p.data) * \
                                       (1-1/(group['gamma']*state['step']+1))

                # proper weight decay
                # if group['weight_decay'] != 0:
                #     step_size.add_(group['weight_decay'], p.data)
                    
                p.data.add_(-step_size)

        return loss

class CrossBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(CrossBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(CrossBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['feature_step'] = torch.zeros_like(p.data) 
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # old = torch.sign(exp_avg)
                state['feature_step'].add_(crossed_zero(exp_avg,
                                                        exp_avg * beta1 + \
                                                        (1 - beta1) * grad))

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # state['feature_step'] += (old * exp_avg <= 0).float()
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['feature_step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['feature_step']))
                step_size = torch.max(torch.min(step_size / denom, upper_bound),
                                      lower_bound) * exp_avg
                # step_size = torch.full_like(denom, step_size)
                # step_size.div_(denom)
                # step_size = torch.max(torch.min(step_size, upper_bound), lower_bound)
                # step_size.mul_(exp_avg)
                
                # step_size = torch.full_like(denom, step_size)
                # step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                # proper weight decay
                # if group['weight_decay'] != 0:
                #     step_size.add_(group['weight_decay'], p.data)
                
                p.data.add_(-step_size) 
                state['alpha_ratio'] = 1-1/(group['gamma']*state['feature_step']+1)
                
        return loss

class CrossAdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 wait=10, eps=1e-8, weight_decay=0, amsbound=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound, wait=wait)
        super(CrossAdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(CrossAdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['feature_step'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                    state['since_crossed_zero'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                direction_crossed_zero = crossed_zero(exp_avg,
                                                      exp_avg * beta1 + \
                                                      (1 - beta1) * grad)
                state['feature_step'].add_(direction_crossed_zero)
                state['since_crossed_zero'].add_(1).mul_(1 - direction_crossed_zero)
                forget = (state['since_crossed_zero'] >= group['wait']).float()
                state['feature_step'].add_(-forget)
                state['since_crossed_zero'].mul_(1 - forget) 
                state['feature_step'] = torch.max(torch.zeros_like(p.data),
                                                  state['feature_step'])

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1-1/(group['gamma']*state['feature_step']+1))
                upper_bound = final_lr * (1+1/(group['gamma'] * state['feature_step']))

                step_size = torch.max(torch.min(step_size / denom, upper_bound),
                                      lower_bound) * exp_avg
                # step_size = torch.full_like(denom, step_size)
                # step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                # how much sgd it is
                state['alpha_ratio'] = 1-1/(group['gamma']*state['feature_step']+1)

                # proper weight decay
                if group['weight_decay'] != 0:
                    step_size.add_(group['weight_decay'], p.data)
                
                p.data.add_(-step_size)

        return loss
    
class CrossVarSGD(Optimizer):
    '''
    use whether momentum cross zero to adjust learning rate
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 final_lr=0.1, gamma=1e-3,
                 eps=1e-8, max_lr=None):

        final_lr = lr / final_lr        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr,
                        final_lr=final_lr, gamma=gamma)
        super(CrossVarSGD, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))        

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['feature_step'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(g2)'] = torch.zeros_like(p.data)
                    state['EMA(w2)'] = torch.zeros_like(p.data)                    
                    state['var(w)'] = torch.zeros_like(p.data)                    
                    state['var(g)'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                state['step'] += 1
                state['feature_step'] += crossed_zero(state['EMA(g)'],
                                                      state['EMA(g)'] * beta1 + \
                                                      (1 - beta1) * grad)
                

                # Decay the first and second moment running average coefficient
                # calculate variance
                dg = grad - state['EMA(g2)']
                state['var(g)'] = beta2 * (state['var(g)'] + (1-beta2) * dg**2)
                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)
                state['EMA(g2)'].mul_(beta2).add_(1-beta2, grad)

                dw = p.data - state['EMA(w2)']
                state['var(w)'] = beta2 * (state['var(w)'] + (1-beta2) * dw**2)
                state['EMA(w2)'].mul_(beta2).add_(1-beta2, p.data)      

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_g2 = state['EMA(g2)'] / bias_correction2
                var_g = state['var(g)'] / bias_correction2
                var_w = state['var(w)'] / bias_correction2

                curvature = var_g

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / \
                                          (group['gamma'] * state['feature_step'] + 1))

                denom = curvature.sqrt().add_(lower_bound + group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = lower_bound / torch.sqrt(curvature)
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                

        return loss

class CrossAdaSGD(Optimizer):
    '''
    add Alpha term to SGD with variance normalization
    effective learning rate can be both high and low
    by dynamically adjust feature steps
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 final_lr=0.1, gamma=1e-3, wait=100, # wait steps before decrease eps
                 eps=1e-8):

        final_lr = lr / final_lr
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, wait=wait, eps=eps,
                        final_lr=final_lr, gamma=gamma)
        super(CrossAdaSGD, self).__init__(params, defaults)
        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))        

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['feature_step'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state['EMA(g)'] = torch.zeros_like(p.data)
                    state['EMA(g2)'] = torch.zeros_like(p.data)
                    state['var(g)'] = torch.zeros_like(p.data)
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['since_crossed_zero'] = torch.zeros_like(p.data)

                beta1, beta2 = group['betas']
                state['step'] += 1
                direction_crossed_zero = crossed_zero(state['EMA(g)'],
                                                      state['EMA(g)'] * beta1 + \
                                                      (1 - beta1) * grad)
                state['feature_step'].add_(direction_crossed_zero)
                state['since_crossed_zero'].add_(1).mul_(1 - direction_crossed_zero)
                forget = (state['since_crossed_zero'] >= group['wait']).float()
                state['feature_step'].add_(-forget)
                state['feature_step'] = torch.max(torch.zeros_like(p.data),
                                                  state['feature_step'])
                
                # Decay the first and second moment running average coefficient
                # calculate variance
                dg = grad - state['EMA(g2)']
                state['var(g)'] = beta2 * (state['var(g)'] + (1-beta2) * dg**2)
                state['EMA(g)'].mul_(beta1).add_(1-beta1, grad)
                state['EMA(g2)'].mul_(beta2).add_(1-beta2, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                E_g = state['EMA(g)'] / bias_correction1
                E_g2 = state['EMA(g2)'] / bias_correction2
                var_g = state['var(g)'] / bias_correction2

                curvature = var_g

                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / \
                                          (group['gamma'] * state['feature_step'] + 1))

                denom = curvature.sqrt().add_(lower_bound + group['eps'])
                # for book keeping purpose
                state['alpha_ratio'] = (lower_bound+group['eps']) / torch.sqrt(curvature)
                
                p.data.addcdiv_(-group['lr'], E_g, denom)
                state['last_grad'] = grad.data.clone()

        return loss
    
class Swats(Optimizer):
    '''
    switch from Adam to SGD https://arxiv.org/pdf/1712.07628.pdf
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-9, weight_decay=0,
                 amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(Swats, self).__init__(params, defaults)
        self.SGD = False # SGD phase or not
        self.sgd_time = 0

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.SGD: self.sgd_time += 1
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                amsgrad = group['amsgrad']
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['sgd_m'] = torch.zeros_like(p.data)
                    state['sgd_lr'] = 0
                    state['^'] = 0 # learning rate in sgd phase
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                if self.SGD:
                    state['sgd_m'].mul_(beta1).add_(grad) # note no 1-beta1
                    p.data.add_(-(1-beta1) * state['^'], state['sgd_m'])
                    state['alpha_ratio'] = torch.ones_like(p.data)
                    continue

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                pk = -step_size * exp_avg / denom
                p.data.add_(pk)
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['alpha_ratio'] = torch.zeros_like(p.data)

                pk = pk.view(-1)
                grad = grad.view(-1)
                if pk.dot(grad) != 0:
                    step_size = - pk.dot(pk) / pk.dot(grad)
                    state['sgd_lr'] = state['sgd_lr'] * beta2 + (1-beta2) * step_size
                    lr_stability = state['sgd_lr'] / bias_correction2 - step_size
                    if state['step'] > 1 and \
                       torch.abs(lr_stability) < group['eps']:
                        self.SGD = True
                        state['^'] = state['sgd_lr'] / bias_correction2
        return loss

class SGD_adjust():
    ''' 
    warm is number of periods to warm up to div * lr from lr with linear rate
    '''
    def __init__(self, opt, div=4, warm=500, patience=10):
        if 'SGD' not in opt.__class__.__name__:
            raise ValueError("Invalid optimizer, must be SGD variants")
        if div < 2:
            raise ValueError("Divided number must be >= 2: now is {}".format(div))
        
        self.opt = opt
        self.div = div
        self.warm = warm
        self.patience = patience

        self.beta = 1 - 1/patience
        self.step_ = 0
        self.n_higher = 0
        self.l_mean = None
        self.new_l_mean = None
        self.l0 = 0
        self.base_lrs = list(map(lambda group: group['lr'], opt.param_groups))

        self.lrs = []
        self.nhighers = []
        self.ls = []

    def diverge(self, l):
        self.step_ += 1
        MAX_L = 10**10

        # if l > MAX_L: assert False, "please lower learning rate"
        if self.new_l_mean is None: self.new_l_mean = l
        else: self.new_l_mean = self.beta * self.new_l_mean + (1-self.beta) * l
        if self.l_mean is not None:
            # if l > self.l_mean:
            #     self.n_higher += 1
            self.n_higher = l - self.l_mean
        else:
            self.l0 += l
        
        if self.step_ % self.patience != 0:
            return False # data collection period

        # if more than half time higher than previous mean, than diverges
        #res = self.n_higher * 2 > self.patience
        res = self.n_higher > 0
        self.nhighers.append(self.n_higher)
        self.ls.append(l)
        
        # reset
        self.n_higher = 0
        self.l_mean = copy.deepcopy(self.new_l_mean)
        return res
        
    def set_lr_mul(self, increase=False):
        for i, group in enumerate(self.opt.param_groups):
            if i == 0: self.lrs.append(group['lr'])
            if increase:
                group['lr'] *= self.div ** (1/self.warm)
            else:
                group['lr'] = max(group['lr']  / self.div, 1e-10)
                self.base_lrs[i] = group['lr']
                self.warm *= 2
        
    def step(self, loss):
        converge =  not self.diverge(loss)
        self.set_lr_mul(increase=converge)
        
######################### learning rate adjusting ########################
class LR_schedule_fixed():

    def __init__(self, lr, schedule, gamma, optimizer, use_cross_zero=True):
        
        self.lr = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                self.lr[p] = torch.ones_like(p.data) * lr

        self.init_lr = lr
        self.gamma = gamma
        self.schedule = schedule
        self.old_grad = {}
        self.feature_step = defaultdict(int)
        self.use_cross_zero = use_cross_zero
        
    def adjust_learning_rate(self, optimizer, epoch):
        
        # update cross_zero
        if self.use_cross_zero:        
            max_step = 0
            min_step = epoch + 1
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    state = optimizer.state[p]
                    assert 'grad' in state and 'lr' in state, "must have grad and lr"  
                    if self.old_grad.get(p) is not None:
                        # print(self.old_grad[p].view(-1)[0], state['grad'].view(-1)[0])
                        self.feature_step[p] += crossed_zero(self.old_grad[p], state['grad'])
                        max_step = max(torch.max(self.feature_step[p]).item(), max_step)
                        min_step = min(torch.min(self.feature_step[p]).item(), min_step)

                    self.old_grad[p] = state['grad'].data.clone()

        # print("max min", max_step, min(min_step, max_step))
        # set learning rate for each parameter separately
        if epoch in self.schedule:
            for group in optimizer.param_groups:

                group['lr'] *= self.gamma
                
                if self.use_cross_zero:                
                    for p in group['params']:
                        if p.grad is None:
                            continue

                        if self.feature_step[p] is 0 or max_step == 0:
                            continue
                        # self.lr[p][self.feature_step[p] > 0] *= self.gamma
                        decay_rate = 1 / (1 + self.feature_step[p] / max_step * \
                                          (1 / self.gamma - 1))
                        self.lr[p] *=  decay_rate

                    state = optimizer.state[p]
                    state['lr'] = self.lr[p]

            if self.use_cross_zero: # refresh
                self.feature_step = defaultdict(int)
                # print("feature step reset to 0")

        # refresh gradient at each epoch
        if self.use_cross_zero:
            reset_grad(optimizer)

class LR_reduce_validation(): # reduce learning rate based on validation

    def __init__(self, lr, gamma, optimizer, use_cross_zero=True, use_max=False,

                 patience=10):
        # use_max signals whether or not higher value is better
        # default false
        self.use_max = use_max
        self.patience = patience
        
        self.lr = {}
        for group in optimizer.param_groups:
            for p in group['params']:
                self.lr[p] = torch.ones_like(p.data) * lr

        self.init_lr = lr
        self.gamma = gamma
        self.old_grad = {}
        self.feature_step = defaultdict(int)
        self.use_cross_zero = use_cross_zero

        self.best_so_far = None
        self.best_streak = 0

    def decay_or_not(self, performance):
        if not self.use_max:
            performance = -performance
            
        if self.best_so_far is None or performance > self.best_so_far:
            #print(self.best_so_far)
            self.best_so_far = performance
            self.best_streak = 0
        else:
            self.best_streak += 1

        decay = self.best_streak >= self.patience
        if decay:
            self.best_streak = 0
        return decay
    
    def adjust_learning_rate(self, optimizer, validation_performance):
        # note: use validation gradient
        # update cross_zero
        max_step = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = optimizer.state[p]
                assert 'grad' in state and 'lr' in state, "must have grad and lr"
                if self.old_grad.get(p) is not None:
                    self.feature_step[p] += crossed_zero(self.old_grad[p], state['grad'])
                    max_step = max(torch.max(self.feature_step[p]).item(), max_step)

                self.old_grad[p] = state['grad'].data.clone()
        
        # set learning rate for each parameter separately
        if self.decay_or_not(validation_performance):
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    if self.use_cross_zero:
                        if self.feature_step[p] is 0 or max_step == 0:
                            continue
                        # self.lr[p][self.feature_step[p] > 0] *= self.gamma
                        decay_rate = 1 / (1 + self.feature_step[p] / max_step * \
                                          (1 / self.gamma - 1))
                        self.lr[p] *=  decay_rate
                    else:
                        self.lr[p] *= self.gamma 
                    state = optimizer.state[p]
                    state['lr'] = self.lr[p]

            if self.use_cross_zero: # refresh
                self.feature_step = defaultdict(int)

        # refresh gradient at each epoch
        reset_grad(optimizer)
                

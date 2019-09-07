import torch

# different explicit regularizations
class L2(object):

    def __init__(self, opt, coef):
        self.opt = opt
        self.coef = coef

    def loss(self):
        l = 0
        for group in self.opt.param_groups:
            for p in group['params']:
                l += torch.sum(p**2)
        return self.coef * l
    

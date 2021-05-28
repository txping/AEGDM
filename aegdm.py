import torch
from torch.optim import Optimizer

class AEGDM(Optimizer):
    r"""Implements AEGDM algorithm.
    It has been proposed in `AEGDM: Adaptive Gradient Decent with Energy and Momentum`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.1)
        c (float, optional): term added to the original objective function (default: 1)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _AEGDM: Adaptive Gradient Decent with Energy and Momentum:
    """

    def __init__(self, params, lr=0.01, c=1.0,
                 momentum=0.9, dampening=0, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, c=c, momentum=momentum,
                        dampening=dampening, weight_decay=weight_decay)

        super(AEGDM, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AEGDM, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """

        # Make sure the closure is defined and always called with grad enabled
        closure = torch.enable_grad()(closure)
        loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            if not 0.0 < loss+group['c']:
                raise ValueError("c={} does not satisfy f(x)+c>0".format(group['c']))

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            c = group['c']

            # Evaluate g(x)=(f(x)+c)^{1/2}
            sqrtloss = torch.sqrt(loss.detach() + c)

            for p in group['params']:
                if p.grad is None:
                    continue
                df = p.grad
                if df.is_sparse:
                    raise RuntimeError('AEGDM does not support sparse gradients')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['energy'] = sqrtloss * torch.ones_like(p)
                    state['buf'] = torch.zeros_like(p)

                energy = state['energy']
                buf = state['buf']

                # Evaluate dg/dx = (df/dx) / (2*g(x))
                dg = df / (2 * sqrtloss)

                # Update energy
                energy.div_(1 + 2 * group['lr'] * dg ** 2)

                # Perform weight decay
                if weight_decay != 0:
                    dg = dg.add(p, alpha=weight_decay)

                if momentum != 0:
                    buf.mul_(momentum).add_(dg, alpha=1 - dampening)
                    dg = buf

                p.addcmul_(energy, dg, value=-2 * group['lr'])

        return loss

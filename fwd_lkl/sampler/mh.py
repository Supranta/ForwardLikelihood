import numpy as np

__all__ = ["sampler"]

class MHSampler(object):
    def __init__(self, lnprob, lnprob_args=[], verbose=False):
        self.lnprob        = lnprob
        self.lnprob_args   = lnprob_args
        self.lnprob_kwargs = None
        self.verbose       = verbose

    def sample_one_step(self, x_old, step_cov, lnprob_kwargs=None):
        if(lnprob_kwargs is not None):
            self.lnprob_kwargs=lnprob_kwargs
            lnprob_old = self.get_lnprob(x_old)
        lnprob_old = self.get_lnprob(x_old)
        x_proposed = np.random.multivariate_normal(mean=x_old, cov=step_cov)

        lnprob_prop = self.get_lnprob(x_proposed)
        diff = lnprob_prop - lnprob_old

        if(self.verbose):
            print("diff: %2.3f"%(diff))

        lnprob0 = lnprob_old
        acc = True
        if(diff > 0.):
            x_new = x_proposed
            lnprob0 = lnprob_prop
        else:
            u = np.random.uniform(0.,1.)
            if(np.log(u) < diff):
                x_new = x_proposed
                lnprob0 = lnprob_prop
            else:
                acc = False
                x_new = x_old
        self.current_lnprob = lnprob0
        return x_new, acc, lnprob0

    def get_lnprob(self, x):
        """Return lnprob at the given position."""
        if(self.lnprob_kwargs is not None):
            return self.lnprob(x, *self.lnprob_args, **self.lnprob_kwargs)
        return self.lnprob(x, *self.lnprob_args)

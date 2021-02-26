import numpy as np

__all__ = ["sampler"]

class HMCSampler(object):
    """
    A class for doing Hamiltonian Monte-Carlo
    :param ndim: Dimension of the parameter space being sampled
    :param psi: The negative of the log-posterior, a.k.a Potential
    :param grad_psi: The gradient of the Potential
    :param Hamiltonian_mass: array of masses for the hamil
    :param psi_args: (optional) extra positional arguments for the function psi
    :param grad_psi_args: (optional) extra positional arguments for the function grad_psi
    """

    def __init__(self, ndim, psi, grad_psi, mass_matrix, verbose=False,
                 psi_args=[], grad_psi_args=[]):
        self.ndim             = ndim
        self.psi              = psi
        self.grad_psi         = grad_psi

        self.mass_matrix   = mass_matrix
        self.M_inv         = np.linalg.inv(mass_matrix)

        self.psi_args         = psi_args
        self.grad_psi_args    = grad_psi_args
        self.psi_kwargs       = None
        self.grad_psi_kwargs  = None
        self.verbose          = verbose

    def sample_one_step(self, x_old, time_step, N_LEAPFROG, psi_kwargs=None, grad_psi_kwargs=None):
        """
        A function to run one step of HMC
        """
        if(psi_kwargs is not None):
            self.psi_kwargs=psi_kwargs
        if(grad_psi_kwargs is not None):
            self.grad_psi_kwargs=grad_psi_kwargs

        p = np.random.multivariate_normal(np.zeros(self.ndim), self.mass_matrix)

        H_old, _, _ = self.Hamiltonian(x_old, p)

        dt = np.random.uniform(0, time_step)
        N = np.random.randint(1,N_LEAPFROG+1)

        x_proposed, p_proposed = self.leapfrog(x_old, p, dt, N)

        H_proposed, KE, _ = self.Hamiltonian(x_proposed, p_proposed)

        diff = H_proposed-H_old

        if(self.verbose):
            print("dt: %2.3f, N:%d"%(dt, N))
            print("diff: %2.3f"%(diff))
        accepted = False
        if(diff < 0.0):
            x_old = x_proposed
            accepted = True
        else:
            rand = np.random.uniform(0.0, 1.0)
            log_rand = np.log(rand)

            if(-log_rand > diff):
                x_old = x_proposed
                accepted = True

        return x_old, -self.get_psi(x_old), accepted, KE

    def leapfrog(self, x, p, dt, N):
        """
        Returns the new position and momenta evolved from the position {x,p} in phase space, with `N_LEAPFROG` leapfrog iterations.
        :param x: The current position in the phase space
        :param p: The current momenta
        :param dt: The time step to which it is evolved
        :param N_LEAPFROG: Number of leapfrog iterations
        """
        x_old       = x
        p_old       = p

        for i in range(N):
            psi_grad = self.get_grad_psi(x_old)

            p_new = p_old-(dt/2)*psi_grad
            x_new = x_old.copy()
            ##=============================================
            x_new += dt * np.matmul(self.M_inv, p_new)
            ##=============================================
            psi_grad = self.get_grad_psi(x_new)
            p_new = p_new-(dt/2)*psi_grad

            x_old, p_old = x_new, p_new

        return x_new, p_new

    def Hamiltonian(self, x, p):
        """
        Returns the hamiltonian for a given position and momenta
        :param x: The position in the parameter space
        :param p: The set of momenta
        """
        KE = 0.5 * np.sum(p * np.matmul(self.M_inv, p))

        PE = self.get_psi(x)
        if(self.verbose):
            print("KE: %2.4f"%(KE))
            print("PE: %2.4f"%(PE))
        H = KE + PE
        return H, KE, PE

    def get_psi(self, x):
        """Return psi at the given position."""
        if(self.psi_kwargs is not None):
            return self.psi(x, *self.psi_args, **self.psi_kwargs)
        return self.psi(x, *self.psi_args)

    def get_grad_psi(self, x):
        """Return grad_psi at the given position."""
        if(self.grad_psi_kwargs is not None):
            return self.grad_psi(x, *self.grad_psi_args, **self.grad_psi_kwargs)
        return self.grad_psi(x, *self.grad_psi_args)

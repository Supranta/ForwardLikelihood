import matplotlib.pyplot as plt

def chain_plot(chain, labels=None, savefig=None, show=True):
    """
    Plots the MCMC chain for the emcee MCMC chain format
    Args:
        chain(array): MCMC Chain in the shape of (N_WALKER, N_MCMC ,N_DIM)
        labels(list): Labels of the plot
    """
    N_DIM = chain.shape[2]
    f, ax = plt.subplots(N_DIM, 1, figsize=(10, 1.75*N_DIM))
    for i in range(N_DIM):
        if i==N_DIM - 1:
            ax[i].set_xlabel('MCMC step')
        if labels is not None:
            ax[i].set_ylabel(labels[i])
        ax[i].plot(chain[:,:,i].T, 'k', linewidth=0.2)
    if savefig is not None:
        plt.savefig(savefig, dpi=200)
    if(show):
        plt.show()

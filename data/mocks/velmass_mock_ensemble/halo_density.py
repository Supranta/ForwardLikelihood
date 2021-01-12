import numpy as np
import sys

mock_num = sys.argv[1]
savedir = 'mock'+mock_num

print("Reading halo catalog...")
halos = np.load(savedir+'/halos.npy')

X = halos['X']
Y = halos['Y']
Z = halos['Z']

N = 256
L = 500.
l = L/N

halo_dens = np.zeros((N,N,N))

ix = (N * (X+250.) / 500.).astype(int)
iy = (N * (Y+250.) / 500.).astype(int)
iz = (N * (Z+250.) / 500.).astype(int)

print(np.min(ix), np.max(ix))
print(np.min(iy), np.max(iy))
print(np.min(iz), np.max(iz))

for i in range(len(halos)):
    if(i%1000==0):
        print(i)
    halo_dens[ix[i], iy[i], iz[i]] += 1

halo_delta = halo_dens / np.mean(halo_dens) - 1.

def Fourier_ks(N_BOX, l):
    kx = 2*np.pi*np.fft.fftfreq(N_BOX,d=l)
    ky = 2*np.pi*np.fft.fftfreq(N_BOX,d=l)
    kz = 2*np.pi*np.fft.fftfreq(N_BOX,d=l)

    N_BOX_Z = (N_BOX//2 +1)

    kx_vec = np.tile(kx[:, None, None], (1, N_BOX, N_BOX_Z))
    ky_vec = np.tile(ky[None, :, None], (N_BOX, 1, N_BOX_Z))
    kz_vec = np.tile(kz[None, None, :N_BOX_Z], (N_BOX, N_BOX, 1))

    k_norm = np.sqrt(kx_vec**2 + ky_vec**2 + kz_vec**2)
    k_norm[(k_norm < 1e-10)] = 1e-15

    return k_norm

def smooth_delta(delta, smooth_R=4.):
    delta_k = np.fft.rfftn(delta)
    k_norm = Fourier_ks(N, l)
    smoothing_func = np.exp(-0.5 * k_norm**2 * smooth_R**2)
    delta_k_smooth = delta_k * smoothing_func
    delta_smooth = np.fft.irfftn(delta_k_smooth)
    return delta_smooth

halo_delta_smooth = smooth_delta(halo_delta)

print(np.min(halo_delta_smooth), np.max(halo_delta_smooth))
print(np.std(halo_delta_smooth))

halo_delta_smooth[(halo_delta_smooth < -1.)] = -0.99

print(np.min(halo_delta_smooth), np.max(halo_delta_smooth))
print(np.std(halo_delta_smooth))

np.save(savedir+'/halo_dens_'+str(N)+'.npy', halo_delta_smooth)

import numpy as np
import h5py as h5
import sys

mocknum = sys.argv[1]
savedir = 'mock'+mocknum

def delta2Vgrid(delta, N, L):
    OmegaM = 0.315

    l = L / N
    dV = l**3
    V = L**3

    kx = 2*np.pi*np.fft.fftfreq(N,d=l)
    ky = 2*np.pi*np.fft.fftfreq(N,d=l)
    kz = 2*np.pi*np.fft.fftfreq(N,d=l)

    N_BOX_Z = (N//2 +1)

    kx_vec = np.tile(kx[:, None, None], (1, N, N_BOX_Z))
    ky_vec = np.tile(ky[None, :, None], (N, 1, N_BOX_Z))
    kz_vec = np.tile(kz[None, None, :N_BOX_Z], (N, N, 1))

    k_norm = np.sqrt(kx_vec**2 + ky_vec**2 + kz_vec**2)
    k_norm[(k_norm < 1e-10)] = 1e-15

    k = np.array([kx_vec, ky_vec, kz_vec])

    l = L / N
    f = OmegaM**0.55

    delta_k = np.fft.rfftn(delta) * dV / V

    J = np.complex(0., 1.)

    R = 4.
    Gaussian_filter = np.exp(-0.5 * k_norm * k_norm * R * R)

    v_kx = J * 100. * f * delta_k * k[0] / k_norm / k_norm
    v_ky = J * 100. * f * delta_k * k[1] / k_norm / k_norm
    v_kz = J * 100. * f * delta_k * k[2] / k_norm / k_norm

    vx = (np.fft.irfftn(v_kx) * V / dV)
    vy = (np.fft.irfftn(v_ky) * V / dV)
    vz = (np.fft.irfftn(v_kz) * V / dV)

    return np.array([vx, vy, vz])

if(int(sys.argv[2])==0):
    dens_type = 'halo'
else:
    dens_type = 'particle'

N = int(sys.argv[3])

if(dens_type=='halo'):
    dens = np.load(savedir+'/halo_dens_'+str(N)+'.npy')
elif(dens_type=='particle'):
    dens = np.load(savedir+'/dens_'+str(N)+'.npy')

vel = delta2Vgrid(dens, N, L=500.)

file_save_name = savedir+'/'+dens_type+str(N)+'.h5'
print("Saving "+file_save_name+" ......")
with h5.File(file_save_name, 'w') as f:
    f['velocity'] = vel
    f['density']  = dens

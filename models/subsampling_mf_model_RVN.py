import torch
from torch import nn
from models.rec_models.unet_model import UnetModel
import data.transforms2 as transforms
from pytorch_nufft.nufft2 import nufft, nufft_adjoint
import numpy as np
import matplotlib.pylab as P
from python_implementation.run_projection import proj_handler
from NKI.direct.nn.recurrentvarnet.recurrentvarnet import RecurrentVarNet
from NKI.direct.data.transforms import fft2 as fft2_direct
from NKI.direct.data.transforms import ifft2 as ifft2_direct
def create_sampling_mask_from_trajectory(trajectories,kshape):
    mask = torch.zeros(kshape)
    for traj in trajectories:
        for coord in traj:
            mask[int(coord[1] + 384//2)][int(coord[0]+144//2)] = 1
    return mask.to(trajectories.device)

class Subsampling_Layer(nn.Module):
    def initilaize_trajectory(self, trajectory_learning, initialization, n_shots):
        # x = torch.zeros(self.num_measurements, 2)
        sampel_per_shot = 3001
        sampel_per_shot = 2 ** 9 + 1
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'cartesian':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            if n_shots < 8:
                raise ValueError
            y_list = [-4, -1, -2, -1, 0, 1, 2, 3]
            if n_shots > 8:
                y_space = int(155 // ((n_shots - 8) / 2))
                y_list = y_list + list(range(4 + y_space, 160, y_space)) + [-i for i in
                                                                            range(4 + y_space, 160, y_space)]
            for i, y in enumerate(y_list):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = y
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                Lx = torch.arange(-144 / 2, 144 / 2, 144 / sampel_per_shot).float()
                Ly = torch.arange(-384 / 2, 384 / 2, 384 / sampel_per_shot).float()
                x[i, :, 0] = Lx * np.cos(theta * i)
                x[i, :, 1] = Ly * np.sin(theta * i)

        elif initialization == 'uniform':
            x = (torch.rand(n_shots, sampel_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(n_shots, sampel_per_shot, 2) * self.res / 6
        else:
            print('Wrong initialization')
        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        return

    def initilaize_trajectories(self, trajectory_learning, initialization, n_shots, num_trajectories):
        # x = torch.zeros(self.num_measurements, 2)
        sampel_per_shot = 3001
        sampel_per_shot = 513
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sampel_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'cartesian':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            if n_shots < 8:
                raise ValueError
            y_list = [-4, -1, -2, -1, 0, 1, 2, 3]
            if n_shots > 8:
                y_space = int(155 // ((n_shots - 8) / 2))
                y_list = y_list + list(range(4 + y_space, 160, y_space)) + [-i for i in
                                                                            range(4 + y_space, 160, y_space)]
            for i, y in enumerate(y_list):
                index = 0
                for j in range(sampel_per_shot):
                    x[i, index, 1] = y
                    x[i, index, 0] = j * 320 / sampel_per_shot - 160
                    index += 1
        elif initialization == 'full':
            x = torch.zeros(344, sampel_per_shot, 2)
            for i, y in enumerate(range(-172, 172)):
                x[i, :, 1] = y
                x[i, :, 0] = torch.linspace(-122, 122, sampel_per_shot)
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sampel_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                Lx = torch.arange(-144 / 2, 144 / 2, 144 / sampel_per_shot).float()
                Ly = torch.arange(-384 / 2, 384 / 2, 384 / sampel_per_shot).float()
                x[i, :, 0] = Lx * np.cos(theta * i)
                x[i, :, 1] = Ly * np.sin(theta * i)
        elif initialization == 'uniform':
            x = (torch.rand(num_trajectories, n_shots, sampel_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(num_trajectories, n_shots, sampel_per_shot, 2) * self.res / 6
        else:
            print('Wrong initialization')

        if initialization != 'uniform' and initialization != 'gaussian' and num_trajectories is not None:
            x = x.squeeze(0).repeat(num_trajectories, 1, 1, 1)

        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        return

    def __init__(self, decimation_rate, res, trajectory_learning, initialization, n_shots, interp_gap, \
                 projection_iterations=10e2, project=False, SNR=False, device='cuda', num_trajectories=None):
        super().__init__()

        self.decimation_rate = decimation_rate
        self.res = res
        self.num_measurements = res ** 2 // decimation_rate
        if num_trajectories is not None:
            self.initilaize_trajectories(trajectory_learning, initialization, n_shots,
                                         num_trajectories=num_trajectories)
        else:
            self.initilaize_trajectory(trajectory_learning, initialization, n_shots)
        self.SNR = SNR
        self.project = project
        self.interp_gap = interp_gap
        self.device = device
        self.iters = projection_iterations

    def forward(self, input):
        # interpolate
        if self.interp_gap > 1:
            assert (len(self.x.shape) == 3 or len(self.x.shape) == 4)
            if len(self.x.shape) == 3:
                t = torch.arange(0, self.x.shape[1], device=self.x.device).float()
                t1 = t[::self.interp_gap]
                x_short = self.x[:, ::self.interp_gap, :]
                if self.project:
                    with torch.no_grad():
                        self.x.data = proj_handler(self.x.data, num_iters=self.iters)
                else:
                    for shot in range(x_short.shape[0]):
                        for d in range(2):
                            self.x.data[shot, :, d] = self.interp(t1, x_short[shot, :, d], t)

                x_full = self.x.reshape(-1, 2)
                input = input.permute(0, 1, 4, 2, 3)
                sub_ksp = nufft(input, x_full, device=self.device)
                img_dr = nufft_adjoint(sub_ksp, x_full, input.shape, device=self.device)
                sub_ksp = transforms.fft2(img_dr.permute(0,1,3,4,2).contiguous())
                return sub_ksp#.permute(0,1,3,2)
                if self.SNR:
                    noise_amp = 0.01
                    noise = noise_amp * torch.randn(sub_ksp.shape)
                    sub_ksp = sub_ksp + noise.to(sub_ksp.device)
                output = nufft_adjoint(sub_ksp, x_full, input.shape, device=self.device)
            elif len(self.x.shape) == 4:
                t = torch.arange(0, self.x.shape[2], device=self.x.device).float()
                t1 = t[::self.interp_gap]
                x_short = self.x[:, :, ::self.interp_gap, :]
                if self.project:
                    with torch.no_grad():
                        self.x.data = proj_handler(self.x.data,num_iters=self.iters)
                else:
                    for frame in range(x_short.shape[0]):
                        for shot in range(x_short.shape[1]):
                            for d in range(2):
                                self.x.data[frame, shot, :, d] = self.interp(t1, x_short[frame, shot, :, d], t)
                output = []
                for frame in range(x_short.shape[0]):
                    x_full = self.x[frame].reshape(-1, 2)
                    curr_input = input[:, frame].permute(0, 3, 1, 2)
                    sub_ksp = nufft(curr_input.unsqueeze(1), x_full, device=self.device)
                    if self.SNR:
                        noise_amp = 0.01
                        noise = noise_amp * torch.randn(sub_ksp.shape)
                        sub_ksp = sub_ksp + noise.to(sub_ksp.device)
                    output.append(nufft_adjoint(sub_ksp, x_full, curr_input.unsqueeze(1).shape, device=self.device))
                output = torch.cat(output, dim=1)

        return output.permute(0, 1, 3, 4, 2)

    def get_trajectory(self):
        return self.x

    def h_poly(self, t):
        tt = [None for _ in range(4)]
        tt[0] = 1
        for i in range(1, 4):
            tt[i] = tt[i - 1] * t
        A = torch.tensor([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]
        ], dtype=tt[-1].dtype)
        return [
            sum(A[i, j] * tt[j] for j in range(4))
            for i in range(4)]

    def interp(self, x, y, xs):
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
        I = P.searchsorted(x[1:].detach().cpu(), xs.detach().cpu())
        dx = (x[I + 1] - x[I])
        hh = self.h_poly((xs - x[I]) / dx)
        return hh[0] * y[I] + hh[1] * m[I] * dx + hh[2] * y[I + 1] + hh[3] * m[I + 1] * dx

    @staticmethod
    def PSNR(im1, im2):
        '''Calculates PSNR between two image signals.
            Args:
                im1 - first image, im2 - second image
            Return:
                scalar - PSNR value
        '''
        fl1 = im1.flatten()
        fl2 = im2.flatten()
        MSE = (((fl1 - fl2) ** 2).sum()) / (fl1.shape[0] * fl2.shape[0])
        R = torch.max(fl1.max(), fl2.max())
        return 10 * torch.log10((R ** 2) / MSE)

    def __repr__(self):
        return f'Subsampling_Layer'


class Subsampling_Model(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, decimation_rate, res,
                 trajectory_learning, initialization, n_shots, interp_gap, multiple_trajectories=False,
                 projection_iters=10e2, project=False, SNR=False, motion = False, device='cuda'):
        super().__init__()
        self.device = device
        self.motion = motion
        if multiple_trajectories:
            self.subsampling = Subsampling_Layer(decimation_rate, res, trajectory_learning, initialization, n_shots,
                                                 interp_gap, projection_iters, project, SNR, device=device,
                                                 num_trajectories=in_chans)
        else:
            self.subsampling = Subsampling_Layer(decimation_rate, res, trajectory_learning, initialization, n_shots,
                                                 interp_gap, projection_iters, project, SNR, device=device)
        self.reconstruction_model = RecurrentVarNet(fft2_direct,ifft2_direct,recurrent_num_layers= 4,recurrent_hidden_channels= 2)#UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)

    def forward(self, input):
        ksp = self.subsampling(input)
        #subsampled_input = transforms.complex_abs(subsampled_input)
        subsample_mask = create_sampling_mask_from_trajectory(self.get_trajectory(), ksp.shape[-3:-1])
        subsample_mask = subsample_mask.repeat(input.shape[0],1,1,1,1).permute(0,1,3,4,2)
        output = torch.clone(ksp)
        for frame in range(ksp.shape[1]):
            output[:,frame,:,:,:] = self.reconstruction_model(ksp[:,frame,:,:,:].unsqueeze(1).contiguous(),subsample_mask.contiguous(),torch.ones_like(ksp[:,frame,:,:,:]).unsqueeze(1).contiguous()).squeeze(1)
        #output = self.reconstruction_model(ksp.unsqueeze(1).contiguous().to(torch.float32),torch.sqrt(torch.sum(self.get_trajectory()**2,dim=-1)).flatten().unsqueeze(-1).contiguous(),torch.ones_like(ksp).unsqueeze(1).contiguous().to(torch.float32)).squeeze(1)
        #image
        #x_full = self.get_trajectory().reshape(-1, 2)
        #output = nufft_adjoint(output.permute(0,1,3,2), x_full, input.permute(0, 1, 4, 2, 3).shape, device=self.device).permute(0,1,3,4,2)
        #output = torch.sqrt(torch.sum(output**2,dim=-1))
        output = transforms.ifft2_regular(output)
        output = transforms.complex_center_crop(output, (output.shape[2:4]))
        return torch.sqrt(torch.sum(output**2,dim=-1)) #+ subsampled_input

    def get_trajectory(self):
        return self.subsampling.get_trajectory()


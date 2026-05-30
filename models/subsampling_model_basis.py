import torch
from torch import nn
from pytorch_nufft.nufft import nufft, nufft_adjoint
import numpy as np
import matplotlib.pylab as P
from WaveformProjection.run_projection import proj_handler
from models.rec_models.ACNN.models.acnn import AcnnModel
from models.rec_models.unet_model import UnetModel
from models.rec_models.vst import SwinTransformer3D as VST
from models.rec_models.vst import BasicLayer as VSTB
from data.transforms import complex_abs
from models.rec_models.vst_unet import VSTUnetModel




class Subsampling_Layer(nn.Module):
    def initilaize_trajectory(self, trajectory_learning, initialization, n_shots):

        sample_per_shot = 2 ** 9 + 1 #must be some 2^n + 1 for some integer n. Here 9 was chosen for 513 points per shot
        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sample_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sample_per_shot - 160
                    index += 1
        elif initialization == 'cartesian':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            if n_shots < 8:
                raise ValueError
            y_list = [-4, -1, -2, -1, 0, 1, 2, 3]
            if n_shots > 8:
                y_space = int(155 // ((n_shots - 8) / 2))
                y_list = y_list + list(range(4 + y_space, 160, y_space)) + [-i for i in
                                                                            range(4 + y_space, 160, y_space)]
            for i, y in enumerate(y_list):
                index = 0
                for j in range(sample_per_shot):
                    x[i, index, 1] = y
                    x[i, index, 0] = j * 320 / sample_per_shot - 160
                    index += 1
        elif initialization == 'radial':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                Lx = torch.arange(-144 / 2, 144 / 2, 144 / sample_per_shot).float()
                Ly = torch.arange(-384 / 2, 384 / 2, 384 / sample_per_shot).float()
                x[i, :, 0] = Lx * np.cos(theta * i)
                x[i, :, 1] = Ly * np.sin(theta * i)
        elif initialization == 'golden':
            '''Credit: This code is based on the implementation from "https://github.com/CARDIAL-nyu/cmr-playground" '''
            golden_ratio = (np.sqrt(5.0) + 1.0) / 2.0
            golden_angle = np.pi / golden_ratio

            radian = np.mod(np.arange(0, n_shots) * golden_angle, 2. * np.pi)
            rho = np.arange(-np.floor(sample_per_shot / 2), np.floor(sample_per_shot / 2)) + 0.5

            _sin = np.sin(radian)
            _cos = np.cos(radian)

            # Complex trajectory
            x = torch.Tensor(np.stack(((rho[..., np.newaxis] * _sin[np.newaxis, ...]),
                                       (rho[..., np.newaxis] * _cos[np.newaxis, ...])), axis=2).transpose((1, 0, 2)))

            # Reshape into (n_spokes, n_readout, 2)
            x = x.reshape((-1, sample_per_shot - 1, 2))

        elif initialization == 'uniform':
            x = (torch.rand(n_shots, sample_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(n_shots, sample_per_shot, 2) * self.res / 6
        else:
            print('Wrong initialization')
        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        self.curr_frame = 0

        return
    
    def initilaize_trajectories(self, trajectory_learning, initialization, n_shots, num_trajectories):
        # x = torch.zeros(self.num_measurements, 2)
        sample_per_shot = 2**9+1

        if initialization == 'spiral':
            x = np.load(f'spiral/{n_shots}int_spiral_low.npy')
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'spiral_high':
            x = np.load(f'spiral/{n_shots}int_spiral.npy') * 10
            x = torch.tensor(x[:, :sample_per_shot, :]).float()
        elif initialization == 'EPI':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            v_space = self.res // n_shots
            for i in range(n_shots):
                index = 0
                for j in range(sample_per_shot):
                    x[i, index, 1] = (i + 0.5) * v_space - 160
                    x[i, index, 0] = j * 320 / sample_per_shot - 160
                    index += 1
        elif initialization == 'cartesian':
            x = torch.zeros(n_shots, sample_per_shot, 2)
            if n_shots < 8:
                raise ValueError
            y_list = [-4, -1, -2, -1, 0, 1, 2, 3]
            if n_shots > 8:
                y_space = int(155 // ((n_shots - 8) / 2))
                y_list = y_list + list(range(4 + y_space, 160, y_space)) + [-i for i in
                                                                            range(4 + y_space, 160, y_space)]
            for i, y in enumerate(y_list):
                index = 0
                for j in range(sample_per_shot):
                    x[i, index, 1] = y
                    x[i, index, 0] = j * 320 / sample_per_shot - 160
                    index += 1
        elif initialization == 'full':
            x = torch.zeros(344, sample_per_shot, 2)
            for i, y in enumerate(range(-172, 172)):
                x[i, :, 1] = y
                x[i, :, 0] = torch.linspace(-122, 122, sample_per_shot)
        elif initialization == 'radial':
            #orig init
            x = torch.zeros(n_shots, sample_per_shot, 2)
            theta = np.pi / n_shots
            for i in range(n_shots):
                Lx = torch.arange(-144 / 2, 144 / 2, 144 / sample_per_shot).float()
                Ly = torch.arange(-384 / 2, 384 / 2, 384 / sample_per_shot).float()
                x[i, :, 0] = Lx * np.cos(theta * i)
                x[i, :, 1] = Ly * np.sin(theta * i)


        elif initialization == 'uniform':
            x = (torch.rand(num_trajectories, n_shots, sample_per_shot, 2) - 0.5) * self.res
        elif initialization == 'gaussian':
            x = torch.randn(num_trajectories, n_shots, sample_per_shot, 2) * self.res / 6
        elif initialization == "golden":
            '''Credit: This code is based on the implementation from "https://github.com/CARDIAL-nyu/cmr-playground" '''
            golden_ratio = (np.sqrt(5.0) + 1.0) / 2.0
            golden_angle = np.pi / golden_ratio

            radian = np.mod(np.arange(0, num_trajectories*n_shots) * golden_angle, 2. * np.pi)
            rho = np.arange(-np.floor(sample_per_shot / 2), np.floor(sample_per_shot / 2)) + 0.5

            _sin = np.sin(radian)
            _cos = np.cos(radian)

            # Complex trajectory
            x = torch.Tensor(np.stack(((rho[..., np.newaxis] * _sin[np.newaxis, ...]),
                             (rho[..., np.newaxis] * _cos[np.newaxis, ...])), axis=2).transpose((1, 0, 2)))


            # Reshape into (n_spokes, n_readout, 2)
            x = x.reshape((num_trajectories,-1,sample_per_shot-1,2))


        else:
            print('Wrong initialization')

        if initialization not in ['uniform', 'gaussian','golden'] and num_trajectories is not None:
            x = x.squeeze(0).repeat(num_trajectories, 1, 1, 1)

        self.x = torch.nn.Parameter(x, requires_grad=bool(int(trajectory_learning)))
        self.curr_frame = 0



        return
    
    def basis2curve(self):
        return torch.sum(self.amplitudes*(self.basis.to(self.amplitudes.device)), dim=2).permute(3,0,1,2)
    
    def __init__(self, decimation_rate, res, trajectory_learning, initialization, n_shots, interp_gap, \
                 projection_iterations=10e2, project=False, SNR=False, device='cuda' if torch.cuda.is_available() else 'cpu', num_trajectories=None):
        super().__init__()
        basis_size=10
        
      
        phi_dot_freqs = torch.arange(0,2*torch.pi,2*torch.pi/basis_size)
        self.decimation_rate = decimation_rate
        self.res = res
        self.num_measurements = res ** 2 // decimation_rate
        self.basis = (torch.cos(phi_dot_freqs)).unsqueeze(0).unsqueeze(0)\
                    .repeat(n_shots,513,8,1).permute(0,1,3,2).to(device).to(torch.float32).unsqueeze(-2) 
        
                
        self.amplitudes = torch.zeros(n_shots,513, basis_size, 2, 1, device=device)
        
        
        #if num_trajectories is not None:
        #    self.initilaize_trajectories(trajectory_learning, initialization, n_shots,
        #                                 num_trajectories=num_trajectories)
        #else:
        #    self.initilaize_trajectory(trajectory_learning, initialization, n_shots)
        self.SNR = SNR
        self.project = project
        self.interp_gap = interp_gap
        self.device = device
        self.iters = projection_iterations
        
        #rad init
        theta = np.pi / n_shots
        for i in range(n_shots):
            Lx = torch.arange(-144 / 2, 144 / 2, 144 / 513).float()
            Ly = torch.arange(-384 / 2, 384 / 2, 384 / 513).float()
            self.amplitudes[i, :, 0, 0,0] = Lx * np.cos(theta * i)
            self.amplitudes[i, :, 0, 1,0] = Ly * np.sin(theta * i)
        
        
      
        self.amplitudes = nn.Parameter(self.amplitudes, requires_grad=bool(int(trajectory_learning)))
       
        self.x = self.basis2curve()
        
       
        
    def forward(self, input):
        # interpolate
        self.x = self.basis2curve()
        
        from matplotlib import pyplot as plt
        import os
        os.makedirs("traj_plot",exist_ok=True)
        fig = plt.figure(figsize=[10, 10])
        for frame in range(8):
            ax = fig.add_subplot(2, 4, frame + 1)
            for i in range(16):
                ax.plot(self.x[frame, i, :, 0].cpu().detach(), self.x[frame, i, :, 1].cpu().detach())
                ax.axis([-165, 165, -165, 165])
        plt.savefig(f"./traj_plot/a.png")
        plt.close()
        fig = plt.figure(figsize=[10, 10])
        for frame in range(8):
            ax = fig.add_subplot(2, 4, frame + 1)
            for i in range(16):
          
                ax.plot(range(8),self.x[:, i, 513//(frame+2), 0].cpu().detach())
                ax.axis([-8, 8, -8, 8])
                break
        plt.savefig(f"./traj_plot/x.png")
        plt.close()
        
        fig = plt.figure(figsize=[10, 10])
        for frame in range(8):
            ax = fig.add_subplot(2, 4, frame + 1)
            for i in range(16):
                ax.plot(range(8),self.x[:, i, 513//(frame+2), 1].cpu().detach())
                ax.axis([-8, 8, -8, 8])
                break
        plt.savefig(f"./traj_plot/y.png")
        plt.close()
        
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
                    output.append(nufft_adjoint(sub_ksp, x_full, curr_input.unsqueeze(1).shape, device=self.device))

                output = torch.cat(output, dim=1)


        return output.permute(0, 1, 3, 4, 2)

    def get_trajectory(self):
        return self.basis2curve()

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
                 projection_iters=10e2, project=False, SNR=False, motion = False, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
        slice_num = 1
        self.reconstruction_model = AcnnModel(in_chans * 2 * slice_num, out_chans * 2 * slice_num, 64, 4, drop_prob, slice_num, False)#UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)

    def forward(self, input):
        subsampled_input = self.subsampling(input)


        output = self.reconstruction_model(subsampled_input)[0]

        return output #+ transforms.complex_abs(subsampled_input)

    def get_trajectory(self):
        return self.subsampling.get_trajectory()


class FeatureVSTB(nn.Module):
    def __init__(self, in_chans, layer_num,vst_depth, feature_num):
        super().__init__()
        self.feature_num = feature_num
        self.in_chans = in_chans
        self.feature_ext = nn.Conv2d(1,feature_num,(3,3),padding="same")
        self.vst = nn.Sequential(*[VSTB(dim=self.feature_num,depth=vst_depth,num_heads=1,window_size=(2,4,4))]*layer_num)
        self.feature_down = nn.Conv2d(feature_num,1,(3,3),padding="same")
    
    def forward(self,x):
        B,_,T,H,W = x.shape
        
        
        x = self.feature_ext(x.permute(0,2,1,3,4).reshape(B*T,1,H,W)).reshape(B,self.feature_num,T,H,W)
        x = self.vst(x)
        
        out = self.feature_down(x.permute(0,2,1,3,4).reshape(B*T,self.feature_num,H,W)).reshape(B,1,T,H,W)
        return out
        

class FeatureVSTB3D(nn.Module):
    def __init__(self, in_chans, layer_num,vst_depth, feature_num):
        super().__init__()
        self.feature_num = feature_num
        self.in_chans = in_chans
        
        self.feature_ext = nn.Conv3d(in_chans,feature_num,(3,3,3),padding="same")
        self.vst = nn.Sequential(*[VSTB(dim=self.feature_num,depth=vst_depth,num_heads=1,window_size=(2,4,4))]*layer_num)
        self.feature_down = nn.Conv3d(feature_num,in_chans,(3,3,3),padding="same")
    
    def forward(self,x):
        B,_,T,H,W = x.shape
        
        
        x = self.feature_ext(x.permute(0,2,1,3,4))
        x = self.vst(x)
        
        out = self.feature_down(x).squeeze()
        return out
        
class FeatureVSTBInterleaved(nn.Module):
    def __init__(self, in_chans, layer_num,vst_depth, feature_num):
        super().__init__()
        self.feature_num = feature_num
        self.in_chans = in_chans
        layers = []
        features = [in_chans,64, 128, 64, in_chans]
        for i in range(len(features)-1):
            layers.append(nn.Conv3d(features[i],features[i+1],(3,3,3),padding="same").cuda())
            layers += [VSTB(dim=features[i+1],depth=vst_depth,num_heads=1,window_size=(2,4,4)).cuda()]*layer_num
            #layers += [nn.ReLU()]
            #self.layers.append(nn.Conv3d(feature_num,in_chans,(3,3,3),padding="same"))
        self.net = nn.Sequential(*layers)
        
    def forward(self,x):
       
        out = self.net(x.permute(0,2,1,3,4)).squeeze()
        
        
        return out

class FeatureVSTB3DFeatures(nn.Module):
    def __init__(self, in_chans, layer_num,vst_depth, feature_num):
        super().__init__()
        self.feature_num = feature_num
        self.in_chans = in_chans
        self.feature_ext = nn.Conv3d(1,feature_num,(3,3,3),padding="same")
        self.vst = nn.Sequential(*[VSTB(dim=self.feature_num,depth=vst_depth,num_heads=1,window_size=(2,4,4))]*layer_num)
        self.feature_down = nn.Conv3d(feature_num,1,(3,3,3),padding="same")
    
    def forward(self,x):
        B,_,T,H,W = x.shape
        
        
        x = self.feature_ext(x)
        x = self.vst(x)
        
        out = self.feature_down(x).squeeze()
        return out
        
class FeatureVST(nn.Module):
    def __init__(self, in_chans, layer_num,vst_depth, feature_num):
        super().__init__()
        self.feature_num = feature_num
        self.in_chans = in_chans
        #self.feature_ext = nn.Conv2d(1,feature_num,(3,3),padding="same")
        self.vst = VST(in_chans=1,depths=[vst_depth]*layer_num,out_channels=in_chans)
   
        self.up_net = nn.Conv2d(24,8,(3,3),padding="same")
    
    def forward(self,x):
        B,_,T,H,W = x.shape
        
        
      
        x = self.vst(x).unsqueeze(1)
        
        out = self.up_net(x.squeeze())
  
        return out

class Subsampling_Model_VST(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_layers, depth, drop_prob, decimation_rate, res,
                 trajectory_learning, initialization, n_shots, interp_gap, multiple_trajectories=False,
                 projection_iters=10e2, project=False, SNR=False, motion = False, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
        slice_num = 1
      
     
        self.reconstruction_model = FeatureVSTBInterleaved(in_chans = in_chans, layer_num = num_layers,vst_depth = depth, feature_num = chans)#FeatureVSTB3D(in_chans = 8, layer_num = num_layers,vst_depth = depth, feature_num = chans)#FeatureVSTB(8,1,1,24)#FeatureVSTB3DFeatures(8,1,1,24)#FeatureVSTB3D(8,1,1,96)#FeatureVST(8,1,1,96)#FeatureVSTB(8,1,1,96)#torch.nn.Sequential(*[VSTB(dim=1,depth=1,num_heads=1,window_size=(2,4,4))]*1)#VST(in_chans=1,depths=[1],out_channels=out_chans)#AcnnModel(in_chans * 2 * slice_num, out_chans * 2 * slice_num, 64, 4, drop_prob, slice_num, False)#UnetModel(in_chans, out_chans, chans, num_pool_layers, drop_prob)
        #self.upsample_net = UnetModel(24,8,32,4,0.0)
        #self.reconstruction_model = VSTUnetModel(8,8,32,4,0.0)
        #self.feature_ext = nn.Conv3d(8,feature_num,(1,3,3),padding="same")
        #self.feature_down = nn.Conv3d(feature_num,8,(1,3,3),padding="same")
        
    def forward(self, input):
        subsampled_input = self.subsampling(input)
        subsampled_input = complex_abs(subsampled_input).unsqueeze(1) #align input to shape [B,C,T,H,W]
        
  
       
        #featured = self.feature_ext(subsampled_input.permute(0,2,1,3,4)).permute(0,2,1,3,4)
        #output = self.reconstruction_model(subsampled_input).squeeze()#self.vst_unet(subsampled_input)#self.reconstruction_model(subsampled_input).squeeze()
        #output = self.feature_down(output.permute(0,2,1,3,4)).permute(0,2,1,3,4).squeeze()
        output = self.reconstruction_model(subsampled_input).squeeze()
        #return self.upsample_net(output)+ subsampled_input.squeeze()
        #output = self.upsample_net(output.reshape(subsampled_input.shape[0],-1,*subsampled_input.shape[-2:]))
        #output = output.reshape(subsampled_input.shape[0],-1,subsampled_input.shape[2],*subsampled_input.shape[-2:]).mean(1)
      
        return output #+ subsampled_input.squeeze() #+ transforms.complex_abs(subsampled_input)

    def get_trajectory(self):
        return self.subsampling.get_trajectory()

import torch
from DRCN.DynamicRadCineMRI.network.reconstruction_network import NUFFTCascade
from DRCN.DynamicRadCineMRI.network.nufft_operator import Dyn2DRadEncObj
class DCRN(NUFFTCascade):
    def __init__(self,unet_model):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super(DCRN, self).__init__(EncObj=None, CNN=unet_model, nu=1,npcg=4,mode='pre_training', use_precon=False, learn_lambda=True)


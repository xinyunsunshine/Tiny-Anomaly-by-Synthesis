import torch
from torch import Tensor
from diffunc.vector_bank import VectorBankBuilder
from featup.util import norm
from diffunc.all import load_input_img,  get_image_paths
import torch.nn.functional as F
import os
import glob
import dataclasses

@dataclasses.dataclass
class ModelParams:
    model_name : str
    patch_size: int
    feature_dim: int

model_params = {
    'dino16': ModelParams(model_name = 'dino16', patch_size = 16, feature_dim = 384),
    'dinov2': ModelParams(model_name = 'dinov2', patch_size = 14, feature_dim = 384),
    'clip': ModelParams(model_name = 'clip', patch_size = 14, feature_dim = 512),
    'maskclip': ModelParams(model_name = 'maskclip', patch_size = 14, feature_dim = 512)
}

class KNN ():
    def __init__(self,
                 max_size: int,
                 upsample_model: str,
                 highres: bool = True,
                 parent_dir: str = '/home/sunsh16e/diffunc/baseline/data/'
                 ) -> None:
        self.max_size = max_size
        self.upsample_model = upsample_model
        self.in_dim = model_params[self.upsample_model].feature_dim

        # size for input into foundation models
        patch_size = model_params[upsample_model].patch_size
        self.new_size = (int(550/patch_size)*patch_size, int(688/patch_size)*patch_size)

        use_norm = (upsample_model!='maskclip') # if 'maskclip', False
        self.upsampler = torch.hub.load("mhamilton723/FeatUp", upsample_model, use_norm = use_norm).to('cuda')
        if not highres:
            self.upsampler = self.upsampler.model
        self.bank_builder = VectorBankBuilder(dim=self.in_dim, max_size=max_size)
    
        self.parent_dir = parent_dir
        self.bank_save_path = os.path.join(parent_dir, 'vector_bank', f'{upsample_model}.pt')
    
    def train(self, train_dataset_path, train_image_paths, verbose = True):
        if glob.glob(self.bank_save_path):
            bank = torch.load(self.bank_save_path)
            with torch.no_grad():
                self.bank_builder.add(bank)
            print('loading bank from cache, size:', self.bank_builder.bank.shape)
            del bank
            torch.cuda.empty_cache()
            return
        count = 0
        for rel_ood_img_path in train_image_paths:
            ori_img_path = os.path.join(train_dataset_path, rel_ood_img_path)
            input = load_input_img(ori_img_path, no_cuda=False)
            self.train_one_image(input)
            if verbose and count % 10 == 0: print(count, rel_ood_img_path)
            count+=1
        bank_save_path = os.path.join(self.parent_dir, 'vector_bank', f'{self.upsample_model}.pt')
        torch.save(self.bank_builder.bank,bank_save_path)
        print('new bank saved, size:', self.bank_builder.bank.shape)

    def train_one_image(self, input):
            
        with torch.no_grad():
            input =  F.interpolate(input, size = self.new_size, mode='bilinear', align_corners=False)
            z_reshaped = (self.upsampler(norm(input)).permute((0, 2, 3, 1 ))).reshape(-1, self.in_dim)  # (B * H * W, Z)
            self.bank_builder.add(z_reshaped)
        del z_reshaped
        torch.cuda.empty_cache()

    def forward(self, input): 
        bank_batch_size = 500
        with torch.no_grad():
            input =  F.interpolate(input, size = self.new_size, mode='bilinear', align_corners=False)
            z = self.upsampler(norm(input)).permute((0, 2, 3, 1 )) #.detach().cpu()  # (B, H, W, Z)
            del input
            torch.cuda.empty_cache()
            # initialize sq_dist to infinity
            sq_dist = torch.inf * torch.ones(z.shape[:-1],
                                                dtype=z.dtype, device=z.device)  # (B, H, W)
            # squared length of z
            sq_norm_z = z.square().sum(dim=-1)  # (B, H, W)
            sq_norm_z = sq_norm_z.unsqueeze(-1)  # (B, H, W, 1)

            for n in range(0, self.bank_builder.bank.shape[0], bank_batch_size):
                # a batch of vectors in the bank
                bn = self.bank_builder.bank[n:n+bank_batch_size, :] #.detach().cpu()  # (BBS, Z)

                # squared length of nth bank batch
                sq_norm_bn = bn.square().sum(dim=-1)  # (BBS,)

                # the dot product between z and nth bank batch, computed
                # efficiently without broadcasting using einsum
                dot_z_bn = torch.einsum('bhwz,nz->bhwn', z, bn)  # (B, H, W, BBS)

                del bn
                torch.cuda.empty_cache()

                # squared distance between z and nth bank batch
                sq_dist_n = sq_norm_z + sq_norm_bn - 2 * dot_z_bn  # (B, H, W, BBS)

                del sq_norm_bn, dot_z_bn
                torch.cuda.empty_cache()

                # minimum distance over the nth bank batch
                sq_dist_n: Tensor = sq_dist_n.min(dim=-1).values  # (B, H, W)

                # update overall minimum
                sq_dist = torch.min(sq_dist, sq_dist_n)  # (B, H, W)

                del sq_dist_n
                torch.cuda.empty_cache()

            del z, sq_norm_z
            torch.cuda.empty_cache()
            distance = sq_dist.clamp(min=0).sqrt()  # (B, H, W)

        return distance

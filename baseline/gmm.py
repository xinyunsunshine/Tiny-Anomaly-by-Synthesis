import torch
from torch import Tensor
from diffunc.vector_bank import VectorBankBuilder
from featup.util import norm
from diffunc.all import load_input_img,  get_image_paths
import torch.nn.functional as F
import os
import glob
import dataclasses
from sklearn.mixture import GaussianMixture
import numpy as np
from typing import Dict, List, Optional, Tuple

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

class GMM ():
    def __init__(self,
                 train_bank_size: int,
                 n_components: int,
                 upsample_model: str,
                 train_max_iter: int,
                 train_n_init: int,
                 baseline: str = 'gmm',
                 highres: bool = False,
                 parent_dir: str = '/home/sunsh16e/diffunc/baseline/data/',
                 ) -> None:
        self.train_bank_size = train_bank_size
        self.upsample_model = upsample_model
        self.in_dim = model_params[self.upsample_model].feature_dim
        self.baseline = baseline

        # size for input into foundation models
        patch_size = model_params[upsample_model].patch_size
        self.new_size = (int(550/patch_size)*patch_size, int(688/patch_size)*patch_size)

        use_norm = (upsample_model!='maskclip') # if 'maskclip', False
        self.upsampler = torch.hub.load("mhamilton723/FeatUp", upsample_model, use_norm = use_norm).to('cuda')
        if not highres:
            self.upsampler = self.upsampler.model
        self.bank_builder = VectorBankBuilder(dim=self.in_dim, max_size=train_bank_size)

        self.K = n_components  # K
        self.gmm = GaussianMixture(n_components=self.K,
                                   covariance_type='full',
                                   max_iter=train_max_iter,
                                   n_init=train_n_init,
                                   verbose=2,
                                   verbose_interval=1)
        # pre-computed quantities
        self.log_det_chol: Optional[Tensor] = None
        self.log_weights: Optional[Tensor] = None
        self.minus_half_Z_log_two_pi: Optional[Tensor] = None

        self.dbl_params = dict(dtype=torch.double, device=torch.cuda.current_device())

        self.parent_dir = parent_dir
        self.bank_save_path = os.path.join(parent_dir, 'vector_bank', f'{upsample_model}.pt')


    def pre_compute(self) -> Tensor:
        """Pre-compute quantities for the GMM using weights, means, and prec_chols."""
        # The determinant of the precision matrix from the Cholesky decomposition
        # corresponds to the negative half of the determinant of the full precision
        # matrix.
        # In short: det(precision_chol) = - det(precision) / 2
        
        # log-det of upper triangular matrices: this is just the log of the product of the diagonal
        self.log_det_chol = torch.sum(
            torch.log(self.prec_chols.reshape(self.K, -1)[:, :: self.in_dim + 1]), 1
        ).to('cuda') # (K,)

        # log weights
        self.log_weights = self.weights.log().to('cuda')  # (K,)

        # -0.5 * log(2 * pi)
        two_pi = torch.tensor(2 * torch.pi, **self.dbl_params)
        self.minus_half_Z_log_two_pi = -0.5 * self.in_dim * torch.log(two_pi).to('cuda')
    
    def train(self, train_dataset_path, train_image_paths, verbose = True):
        if glob.glob(self.bank_save_path):
            bank = torch.load(self.bank_save_path)
            with torch.no_grad():
                self.bank_builder.add(bank)
            print('loading bank from cache, size:', self.bank_builder.bank.shape, self.bank_save_path)
            del bank
            torch.cuda.empty_cache()
            self.fit_gmm()
            return
        
        count = 0
        for rel_ood_img_path in train_image_paths:
            ori_img_path = os.path.join(train_dataset_path, rel_ood_img_path)
            input = load_input_img(ori_img_path, no_cuda=False)
            self.train_one_image(input)
            if verbose and count % 100 == 0: print(count, rel_ood_img_path)
            count+=1
        bank_save_path = os.path.join(self.parent_dir, 'vector_bank', f'{self.upsample_model}.pt')
        torch.save(self.bank_builder.bank,bank_save_path)
        print('new bank saved, size:', self.bank_builder.bank.shape)

        print('start fitting')
        self.fit_gmm()
    
    def train_one_image(self, input):
        with torch.no_grad():
            input =  F.interpolate(input, size = self.new_size, mode='bilinear', align_corners=False)
            z_reshaped = (self.upsampler(norm(input)).permute((0, 2, 3, 1 ))).reshape(-1, self.in_dim)  # (B * H * W, Z)
            self.bank_builder.add(z_reshaped)
        del z_reshaped
        torch.cuda.empty_cache()
        
    
    def fit_gmm(self) -> None:
        """Fit GMM to the training data in bank"""
        param_save_path = os.path.join(self.parent_dir, self.baseline, 'params', f'{self.upsample_model}.pt')
 
        if glob.glob(param_save_path):
            fit_params = torch.load(param_save_path)
            self.weights, self.means, self.prec_chols = fit_params['weights'], fit_params['means'], fit_params['prec_chols']
            self.pre_compute()
            return 
        # bank must be non-empty
        assert self.bank_builder.bank.shape[0] > 0

        print(self.bank_builder)

        # fit GMM
        data: np.ndarray = self.bank_builder.bank.detach().cpu().numpy()  # (N, Z)
        self.gmm.fit(data)

        # retrieve GMM parameters from the fitted model
        def to_tensor(x: np.ndarray) -> Tensor:
            # GMM parameters are in double precision
            return torch.from_numpy(x)
        self.weights = to_tensor(self.gmm.weights_).to('cuda')  # (K,)
        self.means = to_tensor(self.gmm.means_).to('cuda')  # (K, Z)
        self.prec_chols = to_tensor(self.gmm.precisions_cholesky_).to('cuda')  # (K, Z, Z)
        assert np.allclose(self.gmm.weights_.sum(), 1.0)

        # check that the parameters are finite
        assert self.weights.isfinite().all()
        assert self.means.isfinite().all()
        assert self.prec_chols.isfinite().all()

        fit_params = {'weights': self.weights, 'means': self.means, 'prec_chols': self.prec_chols}
    
        torch.save(fit_params, param_save_path)

        # pre-compute quantities for testing
        self.pre_compute()
    
    # def load_registered_buffers_to_sklearn(self) -> None:
    #     """
    #     Load GMM parameters from registered buffers to sklearn's
    #     GaussianMixture
    #     """
    #     self.gmm.weights_ = self.weights.cpu().numpy()
    #     self.gmm.means_ = self.means.cpu().numpy()
    #     self.gmm.precisions_cholesky_ = self.prec_chols.cpu().numpy()

    def forward(self, input):
        with torch.no_grad():
            input =  F.interpolate(input, size = self.new_size, mode='bilinear', align_corners=False)
            z = self.upsampler(norm(input)).permute((0, 2, 3, 1 )) # (B, H, W, Z)
            B, H, W, Z = z.shape
            z_reshaped = z.reshape(-1, self.in_dim) # (B * H * W, Z)
            del z
            torch.cuda.empty_cache()

            log_prob = self.log_prob(z_reshaped)  # (B * H * W,)

        del z_reshaped
        torch.cuda.empty_cache()
        log_prob = log_prob.reshape(1, H, W) * (-1)
        return log_prob



    def log_prob(self, z: Tensor) -> Tensor:
        """
        Directly compute log_prob on GPU
        Code copied from https://github.com/scikit-learn/scikit-learn/blob/7f9bad99d6e0a3e8ddf92a7e5561245224dab102/sklearn/mixture/_gaussian_mixture.py#L423-L427
        , but modified for using torch tensors.

        Args:
            z (Tensor, dtype=float, shape=(N, Z)): tensor of features.
        
        Returns:
            log_prob (Tensor, dtype=float, shape=(N,)): log probabilities under the GMM.
        """
        N, Z = z.shape
        z = z.double()

        log_det = self.log_det_chol  # (K,)
        log_prob: Tensor = torch.empty(N, self.K, **self.dbl_params)  # (N, K)
        for k in range(self.K):
            mu = self.means[k]  # (Z,)
            prec_chol = self.prec_chols[k]  # (Z, Z)
            # np.dot for 2D arrays is equal to matrix multiply
            y = z @ prec_chol - mu @ prec_chol  # (N, Z)
            log_prob[:, k] = torch.sum(torch.square(y), dim=1)
        del z
        torch.cuda.empty_cache()
        # Since we are using the precision of the Cholesky decomposition,
        # `- 0.5 * log_det_precision` becomes `+ log_det_precision_chol`
        # log_prob = -0.5 * (Z * torch.log(2 * torch.pi) + log_prob) + log_det  # (N, K)
        log_prob = -0.5 * log_prob + self.minus_half_Z_log_two_pi + log_det  # (N, K)

        weighted_log_prob = log_prob + self.log_weights  # (N, K)
        final_lob_prob = weighted_log_prob.logsumexp(dim=1)  # (N,)
        return final_lob_prob.float()

import os
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Optional
from tqdm.auto import tqdm

import normflows as nf

from diffunc.all import load_input_img,  get_image_paths
from featup.util import norm

from baseline.gmm import GMM, model_params


class NormalizingFlow (torch.nn.Module):
    def __init__(self,
                 flows: List[nf.flows.Flow],
                 upsample_model: str,
                 gmm: Optional[GMM]=None):

        super().__init__()
        self.upsample_model = upsample_model
        self.in_dim = model_params[self.upsample_model].feature_dim

        # normalizing flow
        q0 = gmm if gmm is not None else nf.distributions.DiagGaussian(shape=self.in_dim, trainable=False)
        self.nf = nf.NormalizingFlow(q0=q0, flows=flows)
    
    def forward(self, z: Tensor) -> Tensor:
        """
        Computes the log-probability of observing the given input, transformed
        by the flow's transforms under the standard Normal distribution.

        Args:
            z (Tensor, dtype=float, shape=(B, C, H, W)): input features.

        Returns:
            log_prob (Tensor, dtype=float, shape=(B, H, W)): log-probability of
                the pixel-wise input features under the normalizing flow.
        """
        B, H, W, C = z.shape
        z = z.reshape(-1, C)  # (B * H * W, C)

        # copied from normflows.NormalizingFlow.log_prob()
        # log_prob: Tensor = self.nf.log_prob(z)  # (B * H * W,)
        log_prob: Tensor = torch.zeros(len(z), dtype=z.dtype, device=z.device)  # (B * H * W,)
        for i in range(len(self.nf.flows) - 1, -1, -1):
            z, log_det = self.nf.flows[i].inverse(z)
            log_prob += log_det

        log_prob += self.nf.q0.log_prob(z)  # (B * H * W,)    
        log_prob = log_prob.reshape(B, H, W)  # (B, H, W)
        return log_prob


class RadialFlow (NormalizingFlow):
    def __init__(self,
                 upsample_model: str,
                 num_layers: int=8,
                 gmm: Optional[GMM]=None):
        
        flows = []
        in_dim = model_params[upsample_model].feature_dim
        for _ in range(num_layers):
            # need to reverse this flow since Radial.inverse() is not implemented
            flow = nf.flows.Reverse(nf.flows.Radial(shape=in_dim))
            flows.append(flow)
        super().__init__(upsample_model=upsample_model, gmm=gmm, flows=flows)

class MAFSlow (NormalizingFlow):
    def __init__(self,
                 upsample_model: str,
                 num_made_transforms: int,
                 num_blocks_per_made: int,
                 hidden_layer_dim: int=None,  # use default 3 * dim + 1
                 use_batch_norm=True,
                 gmm: Optional[GMM]=None):
        
        in_dim = model_params[upsample_model].feature_dim
        if hidden_layer_dim is None:
            hidden_layer_dim = 3 * in_dim + 1
        flows = []
        for _ in range(num_made_transforms):
            flow = nf.flows.affine.MaskedAffineAutoregressive(
                features=in_dim,
                hidden_features=hidden_layer_dim,
                num_blocks=num_blocks_per_made,
                use_batch_norm=use_batch_norm
            )
            flows.append(flow)
        super().__init__(upsample_model=upsample_model, gmm=gmm, flows=flows)

class MAFFast (NormalizingFlow):
    def __init__(self,
                 upsample_model: str,
                 num_made_transforms: int,
                 num_blocks_per_made: int,
                 hidden_layer_dim: int=None,  # use default 3 * dim + 1
                 use_batch_norm=True,
                 gmm: Optional[GMM]=None):
        
        in_dim = model_params[upsample_model].feature_dim
        if hidden_layer_dim is None:
            hidden_layer_dim = 3 * in_dim + 1
        flows = []
        for _ in range(num_made_transforms):
            flow = nf.flows.affine.MaskedAffineAutoregressive(
                features=in_dim,
                hidden_features=hidden_layer_dim,
                num_blocks=num_blocks_per_made,
                use_batch_norm=use_batch_norm
            )
            flow = nf.flows.Reverse(flow)  # reverse flow is much faster!
            flows.append(flow)
        super().__init__(upsample_model=upsample_model, gmm=gmm, flows=flows)

class ResidualFlow (NormalizingFlow):
    def __init__(self,
                 upsample_model: str,
                 num_blocks: int,
                 num_layers_per_block: int,
                 max_lipschitz_iter: int=5,
                 n_samples_power_series: int=1,
                 gmm: Optional[GMM]=None):

        in_dim = model_params[upsample_model].feature_dim
        flows = []
        for _ in range(num_blocks):
            net = nf.nets.LipschitzMLP(channels=[in_dim] * num_layers_per_block,
                                       max_lipschitz_iter=max_lipschitz_iter)
            flow = nf.flows.Residual(net=net,
                                     n_samples=n_samples_power_series,
                                     reduce_memory=False)
            flows.append(flow)
        super().__init__(upsample_model=upsample_model, gmm=gmm, flows=flows)


if __name__ == '__main__':
    train_dataset_path = '/home/sancha/data/RUGD/ddpm_train_sets/full/images/id/train'
    gmm_params_path = '/home/sancha/Desktop/gmm_rugd_maskclip.pt'

    # Training parameters
    num_epochs = 2
    learning_rate = 1e-6

    # Create GMM
    upsample_model = 'maskclip'
    highres = False
    n_components = 20
    gmm = GMM(
        n_components=n_components,
        upsample_model=upsample_model,
        highres=highres,
        train_bank_size=1,
        train_max_iter=0,
        train_n_init=0,
    )

    # Load GMM weights
    fit_params = torch.load(gmm_params_path)
    gmm.weights, gmm.means, gmm.prec_chols = fit_params['weights'], fit_params['means'], fit_params['prec_chols']
    gmm.pre_compute()

    flow = ResidualFlow(
        upsample_model='maskclip',
        gmm=gmm,
        num_blocks=10,
        num_layers_per_block=10,
        max_lipschitz_iter=50,
        n_samples_power_series=10,
    ).cuda()

    # Set model in training mode
    flow.train()

    train_image_paths = get_image_paths(train_dataset_path, nosplit=False)


    optimizer = torch.optim.Adam(flow.parameters(), lr=learning_rate, weight_decay=0)

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch} / {num_epochs}')
        for rel_ood_img_path in train_image_paths:
            ori_img_path = os.path.join(train_dataset_path, rel_ood_img_path)
            input = load_input_img(ori_img_path, no_cuda=False)
            input =  F.interpolate(input, size = gmm.new_size, mode='bilinear', align_corners=False)

            # Run FeatUp
            with torch.no_grad():
                z_train = gmm.upsampler(norm(input)).permute(0, 2, 3, 1 )
        
            log_prob: Tensor = flow(z_train)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            print(f'Loss: {loss.item()}')

        # Save model
        print(f'Saving model to flow_{epoch}.pt ...')
        torch.save(flow.state_dict(), f'flow_{epoch}.pt')

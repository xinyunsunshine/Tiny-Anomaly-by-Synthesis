# Copyright (c) Siddharth Ancha. All rights reserved.

import numpy as np
import torch
from torch import Tensor
from termcolor import colored
from typing import Optional

class VectorBankBuilder:
    """
    Stores a fixed-size sample of vectors without replacement from an online
    stream of vectors, using the reservoir sampling algorithm
    (https://en.wikipedia.org/wiki/Reservoir_sampling), modified for batches
    using np.random.Generator.choice().
    """

    def __init__(self,
                 dim: int,
                 max_size: Optional[int]=None,
                 device: torch.device=torch.cuda.current_device()):
        """
        Args:
            dim (int): dimensionality of the vectors
            max_size (Optional[int]): maximum number of vectors to store. None
                means that there is no limit on the number of vectors to store.
            device (torch.device): the device to store the bank at.
        """
        self.D = dim  # the dimension of the vectors
        self.nMax = max_size  # M is the max size of the bank
        self.device = device

        # random number generator for fast sampling
        self.rng = np.random.default_rng()

        # define nSeen and the bank
        self.clear()

    def clear(self):
        # the number of vectors seen so far
        self.nSeen = 0

        # feature vector bank
        self.bank: Tensor = torch.empty(0, self.D,
                                        dtype=torch.float,
                                        device=self.device)  # (0, D)

    @property
    def nBank(self) -> int:
        """Returns the number of vectors in the bank"""
        return self.bank.shape[0]
    
    def _sample_without_replacement(self,
                                    vectors: Tensor,
                                    nSamples: int) -> Tensor:
        """
        Args:
            vectors (Tensor, shape=(N, D)): a batch of vectors to sample from.
            nSamples (int): number of vectors to sample without replacement.
        """
        if nSamples == 0:
            return torch.empty(0, self.D, dtype=torch.float, device=self.device)

        indices = self.rng.choice(vectors.shape[0],  # total indices
                                  nSamples,  # number of indices we need
                                  replace=False)  # without replacement
        result =  vectors[indices]  # (nSamples, D)
        del vectors
        torch.cuda.empty_cache()
        return result
    
    def add(self, batch: Tensor) -> None:
        """
        batch (Tensor, shape=(nBatch, D)): a batch of vectors to add to bank.
        """
        # check that batch is finite
        if batch.isnan().any():
            raise ValueError(colored("[VBank] Added batch contains NaNs!", 'red'))
        if batch.isinf().any():
            raise ValueError(colored("[VBank] Added batch contains Infs!", 'red'))

        batch = batch.to(device=self.device)  # (nBatch, D)
        nBatch = batch.shape[0]  # number of vectors in the batch
 
        if (self.nMax is None) or (self.nSeen + nBatch <= self.nMax):
            self.bank = torch.cat((self.bank, batch), dim=0)  # (nSeen + nBatch, D)
            self.nSeen += nBatch
            return
        
        # corner case when adding a batch exceeds the max size
        if (self.nSeen < self.nMax) and (self.nSeen + nBatch > self.nMax):
            # split the batch into two
            batch1 = batch[:self.nMax - self.nSeen]  # (nMax - nSeen, D)
            batch2 = batch[self.nMax - self.nSeen:]  # (nBatch - (nMax - nSeen), D)

            self.add(batch1)  # directly append to the bank
            self.add(batch2)  # apply reservoir sampling
            del batch1
            del batch2
            torch.cuda.empty_cache()


        # now the bank is full
        assert self.nBank == self.nMax
        
        # we need to subsample from the union of the bank and the vector batch
        # to simulate a random sample (without replacement) of max_size vectors
        # from the entire stream of vectors seen so far.

        # simulate sampling indices from the entire stream seen so far
        # implementation note:
        #     np.random.choice() is a legacy method. It is extremely slow for
        #     large values of `a`, even if `size` is small. That's because it
        #     creates a full np.random.permutation() under the hood.
        #     np.random.Generator.choice() is much faster. Resources:
        #     - https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html (see note)
        #     - https://ymd_h.gitlab.io/ymd_blog/posts/numpy_random_choice/
        #     - https://stackoverflow.com/a/8505754
        indices = self.rng.choice(self.nSeen + nBatch,  # total indices
                                  self.nMax,  # number of indices we need
                                  replace=False)  # without replacement

        nBankSamples = (indices < self.nSeen).sum()  # number of bank samples
        nBatchSamples = self.nMax - nBankSamples  # number of batch samples

        self.bank = torch.cat(
            (
                self._sample_without_replacement(self.bank, nBankSamples),
                self._sample_without_replacement(batch,     nBatchSamples)
            ),
            dim=0)  # (nMax, D)
        del batch
        torch.cuda.empty_cache()

        self.nSeen += nBatch

    def __str__(self):
        rank = torch.linalg.matrix_rank(self.bank)
        text = colored(
            f"[VBank] Original bank size was {self.nSeen}.\n" \
            f"[VBank] This was subsampled to {self.bank.shape[0]} for training.\n" \
            f"[VBank] Feature dimension is {self.bank.shape[1]}.\n" \
            f"[VBank] Range of randomly subsampled vector bank: [{self.bank.min():.3f}, {self.bank.max():.3f}].\n" \
            f"[VBank] Rank of randomly subsampled vector bank: {rank} / {self.D}.",
        'green')
        if rank < self.D:
            text += colored(f"\n[VBank] ", 'green') + \
                    colored(f"WARNING: Vector bank not full-rank!", 'white', 'on_red', attrs=['bold', 'blink'])
        return text

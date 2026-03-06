"""
Alias method for efficient multinomial sampling.

Implementation based on:
https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
"""
import torch


class AliasMethod(object):
    """
    Alias method for O(1) multinomial sampling.

    The alias method allows sampling from a discrete distribution in O(1) time
    after O(n) preprocessing.
    """

    def __init__(self, probs):
        """
        Initialize alias method sampler.

        Args:
            probs: Probability distribution (will be normalized if sum > 1)
        """
        if probs.sum() > 1:
            probs.div_(probs.sum())

        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0] * K)

        # Sort outcomes into those with probabilities larger and smaller than 1/K
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Create binary mixtures that allocate larger outcomes
        # over the uniform mixture
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        """Move tensors to CUDA."""
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """
        Draw N samples from the multinomial distribution.

        Args:
            N: Number of samples to draw

        Returns:
            Tensor of N samples
        """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(
            0, K
        )
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)

        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj

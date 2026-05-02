import torch


class BatchRenorm(torch.nn.Module):
    def __init__(self, num_features, eps=0.001, momentum=0.01, affine=True, warmup_steps=100_000):
        super().__init__()
        self.register_buffer("ra_mean", torch.zeros(num_features, dtype=torch.float))
        self.register_buffer("ra_var", torch.ones(num_features, dtype=torch.float))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self.scale = torch.nn.Parameter(torch.ones(num_features, dtype=torch.float))
        self.bias = torch.nn.Parameter(torch.zeros(num_features, dtype=torch.float))
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.num_features = num_features
        self.rmax = 3.0
        self.dmax = 5.0
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x):
        raise NotImplementedError()

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0)
            batch_std = (batch_var + self.eps).sqrt()
            if self.steps > self.warmup_steps:
                running_std = (self.ra_var + self.eps).sqrt()
                r = (batch_std / running_std).detach().clamp(1 / self.rmax, self.rmax)
                d = ((batch_mean - self.ra_mean) / running_std).detach().clamp(-self.dmax, self.dmax)
                custom_mean = batch_mean - d * batch_var.sqrt() / r
                custom_var = batch_var / (r**2)
            else:
                custom_mean, custom_var = batch_mean, batch_var
            self.ra_mean += self.momentum * (batch_mean.detach() - self.ra_mean)
            self.ra_var += self.momentum * (batch_var.detach() - self.ra_var)
            self.steps += 1
        else:
            custom_mean, custom_var = self.ra_mean, self.ra_var

        x = (x - custom_mean[None]) / (custom_var[None] + self.eps).sqrt()
        if self.affine:
            x = self.scale * x + self.bias
        return x

    def extra_repr(self):
        return (f"num_features={self.num_features}, momentum={self.momentum}, "
                f"warmup_steps={self.warmup_steps}, affine={self.affine}")


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x):
        if x.dim() == 1:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")

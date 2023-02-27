import gpytorch
import torch

# We will use the simplest form of GP model, exact inference
# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(  gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.RBFKernel(), num_tasks=2, rank=2
        # )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(), num_tasks=2, rank=2
        )

        # print("attr:", dir(self.covar_module.data_covar_module))
        self.covar_module.data_covar_module.lengthscale = torch.tensor(128)
        # mylengthscale = float(config.scale) * np.sqrt(Dim) * np.random.rand(d1,d2)
        # model.covar_module.data_covar_module.kernels[2*i+1].lengthscale = torch.tensor(mylengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # add small noise to the diagonal of the covariance matrix 1e^-6
        covar_x = covar_x.add_jitter(1e-6)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

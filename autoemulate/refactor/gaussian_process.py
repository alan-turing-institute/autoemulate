import gpytorch
import torch

from autoemulate.refactor.base import BaseModel, InputTypeMixin


class GaussianProcessRefactor(gpytorch.models.ExactGP, BaseModel, InputTypeMixin):
    # TODO: can the init method be different across models?
    # Perhaps a match statement to handle the different cases will be suffiocient
    def __init__(self, train_x, train_y, normalize_y=True):
        X, y = self.convert_to_tensor(train_x), self.convert_to_tensor(train_y)
        self.mean_module = None
        self.covar_module = None
        self.normalize_y = normalize_y
        self.n_features_in_ = X.shape[1]
        # Single task for now
        self.n_outputs_ = 1
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # GP's work better when the target values are normalized
        if self.normalize_y:
            self._y_train_mean = y.mean(dim=0)
            self._y_train_std = y.std(dim=0)
            y = (y - self._y_train_mean) / self._y_train_std
        mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([self.n_outputs_])
        )

        # combined RBF + constant kernel works well in a lot of cases
        rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=self.n_features_in_,  # different lengthscale for each feature
            batch_shape=torch.Size([self.n_outputs_]),  # batched multioutput
            # seems to work better when we initialize the lengthscale
        ).initialize(lengthscale=torch.ones(self.n_features_in_) * 1.5)
        constant = gpytorch.kernels.ConstantKernel()
        combined = rbf + constant
        covar_module = gpytorch.kernels.ScaleKernel(
            combined, batch_shape=torch.Size([self.n_outputs_])
        )
        super(GaussianProcessRefactor, self).__init__(X, y, likelihood)
        self.max_epochs = 100
        self.likelihood = likelihood
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        x = self.convert_to_tensor(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def fit(self, X, y):
        X = self.convert_to_tensor(X)
        y = self.convert_to_tensor(y)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        self.train()
        self.likelihood.train()
        for i in range(self.max_epochs):
            optimizer.zero_grad()
            output = self(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.eval()
        return self(X)

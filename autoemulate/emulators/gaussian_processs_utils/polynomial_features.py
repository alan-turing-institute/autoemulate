import itertools
import torch

class PolynomialFeatures:
    def __init__(self, degree):
        self.degree = degree
        self.indices = None

    def fit(self, input_size):
        x = list(range(input_size))

        d = self.degree
        L = []
        while d > 1:
            l = [list(p) for p in itertools.product(x, repeat=d)]
            for li in l:
                li.sort()
            L += list(map(list, np.unique(l, axis=0)))
            d -= 1
        L += [[i] for i in x]

        Ls = []
        for d in range(1, self.degree + 1):
            ld = []
            for l in L:
                if len(l) == d:
                    ld.append(l)
            ld.sort()
            Ls += ld
        self.indices = Ls

    def transform(self, x):
        if not self.indices:
            raise ValueError(
                "self.indices is set to None. Did you forget to call 'fit'?"
            )

        x_ = torch.stack([torch.prod(x[..., i], dim=-1) for i in self.indices], dim=-1)
        return x_
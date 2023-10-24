import numpy as np
import pytest
from numpy.testing import assert_allclose
from ..GPParams import GPParams, CovTransform, CorrTransform
from ..Priors import GPPriors, min_spacing, max_spacing, MeanPriors, WeakPrior, PriorDist
from ..Priors import NormalPrior, LogNormalPrior, GammaPrior, InvGammaPrior
from scipy.stats import norm, gamma, invgamma, lognorm
from scipy.linalg import cho_factor, cho_solve

def test_GPPriors():
    "test the GPPrior class"
    
    gpp = GPPriors(n_corr=4)
    
    assert isinstance(gpp.mean, MeanPriors)
    assert gpp.mean.mean is None
    assert gpp.mean.cov is None
    assert gpp.n_mean == 0
    
    assert gpp.n_corr == 4
    assert isinstance(gpp.cov, WeakPrior)
    assert isinstance(gpp.nugget, WeakPrior)
    assert gpp.nugget_type == "fit"
    
    gpp = GPPriors(corr=[LogNormalPrior(2., 3.), LogNormalPrior(2., 3.)], cov=LogNormalPrior(2., 3.),
                   nugget_type="pivot", mean=MeanPriors([1.], 3.))

    assert isinstance(gpp.mean, MeanPriors)
    assert_allclose(gpp.mean.mean, [1.])
    assert_allclose(gpp.mean.cov, 3.)
    assert gpp.n_mean == 1

    assert all([isinstance(o, LogNormalPrior) for o in gpp.corr])
    assert gpp.n_corr == 2
    assert isinstance(gpp.cov, LogNormalPrior)
    assert gpp.nugget is None
    assert gpp.nugget_type == "pivot"


    gpp = GPPriors(corr=[LogNormalPrior(2., 3.), LogNormalPrior(2., 3.), WeakPrior()], nugget=InvGammaPrior(1., 1.),
                   nugget_type="fixed", mean=(np.array([2., 3.]), np.array([[1., 0.], [0., 1.]])))


    assert isinstance(gpp.corr[0], LogNormalPrior)
    assert isinstance(gpp.corr[1], LogNormalPrior)
    assert isinstance(gpp.corr[2], WeakPrior)

    assert isinstance(gpp.mean, MeanPriors)
    assert_allclose(gpp.mean.mean, [2., 3.])
    assert_allclose(gpp.mean.cov, np.eye(2))

    assert gpp.nugget is None
    
    with pytest.raises(ValueError):
        GPPriors()
        
    with pytest.raises(AssertionError):
        GPPriors(n_corr=1, nugget_type="blah")

def test_GPPriors_mean():
    "test mean functionality of GPPriors"
    
    gpp = GPPriors(n_corr=1)
    
    assert gpp.mean.mean is None
    assert gpp.mean.cov is None
    assert gpp.n_mean == 0
    
    gpp.mean = MeanPriors([2.], 3.)
    assert_allclose(gpp.mean.mean, [2.])
    assert_allclose(gpp.mean.cov, 3.)
    assert gpp.n_mean == 1
    
    gpp.mean = ([3., 4.], [[1., 0.], [0., 1.]])
    assert_allclose(gpp.mean.mean, [3., 4.])
    assert_allclose(gpp.mean.cov, np.eye(2))
    assert gpp.n_mean == 2
    
    gpp.mean = None
    assert gpp.mean.mean is None
    assert gpp.mean.cov is None
    assert gpp.n_mean == 0

    with pytest.raises(ValueError):
        gpp.mean = (1., 2., 3.)

def test_GPPriors_corr():
    "Test correlation priors"
    
    gpp = GPPriors(n_corr=1)
    
    assert gpp.n_corr == 1
    assert len(gpp.corr) == 1
    assert isinstance(gpp.corr[0], WeakPrior)
    
    gpp.corr = [WeakPrior(), LogNormalPrior(1., 1.), WeakPrior()]
    
    assert gpp.n_corr == 3
    assert len(gpp.corr) == 3
    assert isinstance(gpp.corr[0], WeakPrior)
    assert isinstance(gpp.corr[2], WeakPrior)
    assert isinstance(gpp.corr[1], LogNormalPrior)
    
    gpp.corr = None
    assert gpp.n_corr == 3
    assert len(gpp.corr) == 3
    assert all([isinstance(d, WeakPrior) for d in gpp.corr])
    
    with pytest.raises(AssertionError):
        gpp.corr = []
        
    with pytest.raises(TypeError):
        gpp.corr = 1.
        
    with pytest.raises(TypeError):
        gpp.corr = [1., LogNormalPrior(2., 3.)]
        
def test_GPPriors_cov():
    "Test the cov property of GPPriors"
    
    gpp = GPPriors(n_corr=1)
    
    assert isinstance(gpp.cov, WeakPrior)
    
    gpp.cov = InvGammaPrior(2., 3.)
    
    assert isinstance(gpp.cov, InvGammaPrior)
    
    gpp.cov = None
    
    assert isinstance(gpp.cov, WeakPrior)
    
    with pytest.raises(TypeError):
        gpp.cov = 1.

def test_GPPriors_nugget():
    "Test the nugget property of GPPriors"
    
    gpp = GPPriors(n_corr=1)
    
    assert isinstance(gpp.nugget, WeakPrior)
    
    gpp.nugget = InvGammaPrior(2., 3.)
    
    assert isinstance(gpp.nugget, InvGammaPrior)
    
    gpp.nugget = None
    
    assert isinstance(gpp.nugget, WeakPrior)
    
    with pytest.raises(TypeError):
        gpp.nugget = 1.
        
    gpp = GPPriors(n_corr=1, nugget_type="adaptive")
    
    assert gpp.nugget is None
    
    gpp.nugget = InvGammaPrior(2., 3.)
    
    assert gpp.nugget is None
    
    gpp = GPPriors(n_corr=1, nugget_type="pivot")
    
    assert gpp.nugget is None
    
    gpp.nugget = InvGammaPrior(2., 3.)
    
    assert gpp.nugget is None

def test_GPPriors_logp():
    "test the logp method of the GPPrior class"
    
    gpp = GPPriors(corr=[ LogNormalPrior(2., 3.)], cov=GammaPrior(2., 3.), nugget=InvGammaPrior(1., 1.),
                   nugget_type="fit")
                    
    theta = GPParams(n_mean=1, n_corr=1, nugget="fit")
    theta.set_data(np.zeros(3))
    
    logp = gpp.logp(theta)
    
    assert_allclose(logp, np.sum([ float(LogNormalPrior(2., 3.).logp(theta.corr)),
                                   float(GammaPrior(2., 3.).logp(theta.cov)),
                                   float(InvGammaPrior(1., 1.).logp(theta.nugget)) ]))

    gpp = GPPriors(corr=[ LogNormalPrior(2., 3.)], cov=GammaPrior(2., 3.),
                   nugget_type="pivot")
                   
    theta = GPParams(n_mean=1, n_corr=1, nugget="pivot")
    theta.set_data(np.zeros(2))
    
    logp = gpp.logp(theta)
    
    assert_allclose(logp, np.sum([ float(LogNormalPrior(2., 3.).logp(theta.corr)),
                                   float(GammaPrior(2., 3.).logp(theta.cov)) ]))

    with pytest.raises(TypeError):
        gpp.logp(1.)

    theta = GPParams(n_mean=1, n_corr=2, nugget="pivot")
    
    with pytest.raises(AssertionError):
        gpp.logp(theta)
        
    theta = GPParams(n_mean=1, n_corr=1, nugget="fit")
    
    with pytest.raises(AssertionError):
        gpp.logp(theta)
        
    theta = GPParams(n_mean=1, n_corr=1, nugget="pivot")
    
    with pytest.raises(AssertionError):
        gpp.logp(theta)
                      
def test_GPPriors_dlogpdtheta():
    "test the dlogpdtheta method of the GPPriors class"
    
    gpp = GPPriors(corr=[ LogNormalPrior(2., 3.)], cov=GammaPrior(2., 3.), nugget=InvGammaPrior(1., 1.),
                   nugget_type="fit")
                    
    theta = GPParams(n_mean=1, n_corr=1, nugget="fit")
    theta.set_data(np.zeros(3))
    
    partials = gpp.dlogpdtheta(theta)
    
    assert_allclose(partials,
                    [ float(LogNormalPrior(2., 3.).dlogpdx(theta.corr)*CorrTransform.dscaled_draw(theta.corr)),
                      float(GammaPrior(2., 3.).dlogpdx(theta.cov)*CovTransform.dscaled_draw(theta.cov)),
                      float(InvGammaPrior(1., 1.).dlogpdx(theta.nugget)*CovTransform.dscaled_draw(theta.nugget))])

                      
    gpp = GPPriors(corr=[ LogNormalPrior(2., 3.)], cov=GammaPrior(2., 3.),
                   nugget_type="pivot")
                   
    theta = GPParams(n_mean=1, n_corr=1, nugget="pivot")
    theta.set_data(np.zeros(2))
    
    partials = gpp.dlogpdtheta(theta)
    
    assert_allclose(partials,
                    [ float(LogNormalPrior(2., 3.).dlogpdx(theta.corr)*CorrTransform.dscaled_draw(theta.corr)),
                      float(GammaPrior(2., 3.).dlogpdx(theta.cov)*CovTransform.dscaled_draw(theta.cov))])
                      
    with pytest.raises(TypeError):
        gpp.dlogpdtheta(1.)

    theta = GPParams(n_mean=1, n_corr=2, nugget="pivot")
    
    with pytest.raises(AssertionError):
        gpp.dlogpdtheta(theta)
        
    theta = GPParams(n_mean=1, n_corr=1, nugget="fit")
    
    with pytest.raises(AssertionError):
        gpp.dlogpdtheta(theta)
        
    theta = GPParams(n_mean=1, n_corr=1, nugget="pivot")
    
    with pytest.raises(AssertionError):
        gpp.dlogpdtheta(theta)
                      
def test_GPPriors_d2logpdtheta2():
    "test the dlogpdtheta method of the GPPriors class"
    
    gpp = GPPriors(corr=[ LogNormalPrior(2., 3.)], cov=GammaPrior(2., 3.), nugget=InvGammaPrior(1., 1.),
                   nugget_type="fit")
                    
    theta = GPParams(n_mean=1, n_corr=1, nugget="fit")
    theta.set_data(np.zeros(3))
    
    hessian = gpp.d2logpdtheta2(theta)
    
    assert_allclose(hessian,
                    [ float(LogNormalPrior(2., 3.).d2logpdx2(theta.corr)*CorrTransform.dscaled_draw(theta.corr[0])**2
                            + LogNormalPrior(2., 3.).dlogpdx(theta.corr)*CorrTransform.d2scaled_draw2(theta.corr[0])),
                      float(GammaPrior(2., 3.).d2logpdx2(theta.cov)*CovTransform.dscaled_draw(theta.cov)**2
                            + GammaPrior(2., 3.).dlogpdx(theta.cov)*CovTransform.d2scaled_draw2(theta.cov)),
                      float(InvGammaPrior(1., 1.).d2logpdx2(theta.nugget)*CovTransform.dscaled_draw(theta.nugget)**2
                            +InvGammaPrior(1., 1.).dlogpdx(theta.nugget)*CovTransform.d2scaled_draw2(theta.nugget))])
                            
    gpp = GPPriors(corr=[ LogNormalPrior(2., 3.)], cov=GammaPrior(2., 3.),
                   nugget_type="pivot")
                   
    theta = GPParams(n_mean=1, n_corr=1, nugget="pivot")
    theta.set_data(np.zeros(2))
    
    hessian = gpp.d2logpdtheta2(theta)
    
    assert_allclose(hessian,
                    [ float(LogNormalPrior(2., 3.).d2logpdx2(theta.corr)*CorrTransform.dscaled_draw(theta.corr[0])**2
                            + LogNormalPrior(2., 3.).dlogpdx(theta.corr)*CorrTransform.d2scaled_draw2(theta.corr[0])),
                      float(GammaPrior(2., 3.).d2logpdx2(theta.cov)*CovTransform.dscaled_draw(theta.cov)**2
                            + GammaPrior(2., 3.).dlogpdx(theta.cov)*CovTransform.d2scaled_draw2(theta.cov))])


    with pytest.raises(TypeError):
        gpp.d2logpdtheta2(1.)

    theta = GPParams(n_mean=1, n_corr=2, nugget="pivot")
    
    with pytest.raises(AssertionError):
        gpp.d2logpdtheta2(theta)
    
    theta = GPParams(n_mean=1, n_corr=1, nugget="fit")
    
    with pytest.raises(AssertionError):
        gpp.d2logpdtheta2(theta)
        
    theta = GPParams(n_mean=1, n_corr=1, nugget="pivot")
    
    with pytest.raises(AssertionError):
        gpp.d2logpdtheta2(theta)

def test_GPPrior_sample():
    "test the sample method"
    
    np.seterr(all="raise")
    
    gpp = GPPriors(corr=[ LogNormalPrior(2., 3.)], cov=GammaPrior(2., 3.), nugget=InvGammaPrior(1., 1.),
                   nugget_type="fit")

    s = gpp.sample()

    assert len(s) == 3
    
    gpp = GPPriors(mean=([2.], [4.]), n_corr=1)
    
    s = gpp.sample()

    assert len(s) == 3
    assert np.all(s >= -2.5)
    assert np.all(s <=  2.5)

def test_default_prior():
    "test default_prior function"
    
    dist = InvGammaPrior.default_prior(1., 3.)
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(invgamma.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    dist = GammaPrior.default_prior(1., 14.)
    
    assert isinstance(dist, GammaPrior)
    assert_allclose(gamma.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(gamma.cdf(14., dist.shape, scale=dist.scale), 0.995)
    
    dist = LogNormalPrior.default_prior(1., 12.)
    
    assert isinstance(dist, LogNormalPrior)
    assert_allclose(lognorm.cdf(1., dist.shape, scale=dist.scale), 0.005)
    assert_allclose(lognorm.cdf(12., dist.shape, scale=dist.scale), 0.995)
    
    with pytest.raises(ValueError):
        NormalPrior.default_prior(1., 12.)
        
    with pytest.raises(AssertionError):
        InvGammaPrior.default_prior(12., 1.)
        
    with pytest.raises(AssertionError):
        GammaPrior.default_prior(-1., 2.)
        
    with pytest.raises(AssertionError):
        LogNormalPrior.default_prior(1., -2.)
        
    assert isinstance(InvGammaPrior.default_prior(1.e-12, 1.e-11), WeakPrior)

def test_min_spacing():
    "test min_spacing function"
    
    inputs = np.array([1., 2., 4.])
    
    assert_allclose(min_spacing(inputs), np.median([1., 2.]))
    
    np.random.shuffle(inputs)
    
    assert_allclose(min_spacing(inputs), np.median([1., 2.]))
    
    inputs = np.array([[1., 2.], [4., 5.]])
    
    assert_allclose(min_spacing(inputs), np.median([1., 2., 1.]))
    
    assert min_spacing(np.array([1.])) == 0.
    assert min_spacing(np.array([1., 1., 1.])) == 0.
   
def test_max_spacing():
    "text max_spacing function"

    inputs = np.array([1., 2., 4.])

    assert_allclose(max_spacing(inputs), 3.)
    
    np.random.shuffle(inputs)
    
    assert_allclose(max_spacing(inputs), 3.)
    
    inputs = np.array([[1., 2.], [4., 5.]])
    
    assert_allclose(max_spacing(inputs), 4.)
    
    assert max_spacing(np.array([1.])) == 0.
    assert max_spacing(np.array([1., 1., 1.])) == 0.

def test_default_prior_corr():
    "test default_prior_corr"
    
    dist = InvGammaPrior.default_prior_corr(np.array([1., 2., 4.]))
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(invgamma.cdf(np.median([1., 2.]), dist.shape, scale=dist.scale), 0.005)
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    dist = GammaPrior.default_prior_corr(np.array([1., 2., 4.]))
    
    assert isinstance(dist, GammaPrior)
    assert_allclose(gamma.cdf(np.median([1., 2.]), dist.shape, scale=dist.scale), 0.005)
    assert_allclose(gamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    dist = InvGammaPrior.default_prior_corr(np.array([1., 1., 2., 4.]))
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(invgamma.cdf(np.median([1., 2.]), dist.shape, scale=dist.scale), 0.005)
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    assert isinstance(InvGammaPrior.default_prior_corr([1.]), WeakPrior)
    assert isinstance(GammaPrior.default_prior_corr([1., 1., 1.]), WeakPrior)
    assert isinstance(LogNormalPrior.default_prior_corr([1., 2.]), WeakPrior)

def test_default_prior_mode():
    "test default_prior function"
    
    dist = InvGammaPrior.default_prior_mode(1., 3.)
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(np.sqrt(3.), dist.scale/(dist.shape + 1.))
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    with pytest.raises(AttributeError):
        GammaPrior.default_prior_mode(1., 12.)
        
    with pytest.raises(AssertionError):
        InvGammaPrior.default_prior_mode(12., 1.)
        
    with pytest.raises(AssertionError):
        InvGammaPrior.default_prior_mode(-1., 2.)
        
    with pytest.raises(AssertionError):
        InvGammaPrior.default_prior_mode(1., -2.)

def test_default_prior_corr_mode():
    "test default_prior_corr"
    
    dist = InvGammaPrior.default_prior_corr_mode(np.array([1., 2., 4.]))
    
    assert isinstance(dist, InvGammaPrior)
    assert_allclose(np.sqrt(np.median([1., 2.])*3.), dist.scale/(dist.shape + 1.))
    assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)
    
    assert isinstance(InvGammaPrior.default_prior_corr_mode([1.]), WeakPrior)
    assert isinstance(InvGammaPrior.default_prior_corr_mode([1., 1., 1.]), WeakPrior)
    assert isinstance(InvGammaPrior.default_prior_corr_mode([1., 2.]), WeakPrior)

def test_GPPriors_default_priors():
    "test class method creating default priors"
    
    gpp = GPPriors.default_priors(np.array([[1., 4.], [2., 2.], [4., 1.]]),
                                  n_corr=2,
                                  nugget_type="fit")
    
    assert gpp.mean.mean is None
    assert gpp.mean.cov is None
    assert isinstance(gpp.corr[0], InvGammaPrior)
    assert isinstance(gpp.corr[1], InvGammaPrior)
    assert isinstance(gpp.cov, WeakPrior)
    assert isinstance(gpp.nugget, InvGammaPrior)
    
    for dist in gpp.corr:
        assert_allclose(invgamma.cdf(np.median([1., 2.]), dist.shape, scale=dist.scale), 0.005)
        assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)

    assert_allclose(invgamma.cdf(1.e-6, gpp.nugget.shape, scale=gpp.nugget.scale), 0.995)
    assert_allclose(1.e-7, gpp.nugget.scale/(gpp.nugget.shape + 1.))
    
    gpp = GPPriors.default_priors(np.array([[1., 4.], [2., 2.], [4., 1.]]),
                                  n_corr=1,
                                  nugget_type="fit")
    
    assert gpp.mean.mean is None
    assert gpp.mean.cov is None
    assert isinstance(gpp.corr[0], InvGammaPrior)
    assert isinstance(gpp.cov, WeakPrior)
    assert isinstance(gpp.nugget, InvGammaPrior)
    
    for dist in gpp.corr:
        assert_allclose(invgamma.cdf(np.median([1., 2.]), dist.shape, scale=dist.scale), 0.005)
        assert_allclose(invgamma.cdf(3., dist.shape, scale=dist.scale), 0.995)

    assert_allclose(invgamma.cdf(1.e-6, gpp.nugget.shape, scale=gpp.nugget.scale), 0.995)
    assert_allclose(1.e-7, gpp.nugget.scale/(gpp.nugget.shape + 1.))
    
    inputs = np.array([[1.e-8], [1.1e-8], [1.2e-8], [1.3e-8], [1.]])
    gpp = GPPriors.default_priors(inputs, n_corr=1, nugget_type="adaptive")
    
    assert isinstance(gpp.corr[0], InvGammaPrior)
    assert_allclose(invgamma.cdf(max_spacing(inputs), gpp.corr[0].shape, scale=gpp.corr[0].scale), 0.995)
    assert_allclose(np.sqrt(min_spacing(inputs)*max_spacing(inputs)), gpp.corr[0].scale/(gpp.corr[0].shape + 1.))
    assert gpp.nugget is None
    
    inputs = np.array([[1.e-8], [1.1e-8], [1.2e-8], [1.3e-8], [1.]])
    gpp = GPPriors.default_priors(inputs, n_corr=1, nugget_type="pivot")
    
    assert gpp.nugget is None

    with pytest.raises(ValueError):
        gpp = GPPriors.default_priors(np.array([[1., 4.], [2., 2.], [4., 1.]]),
                                      n_corr=3,
                                      nugget_type="fit")

def test_MeanPriors():
    "test the MeanPriors class"
    
    mp = MeanPriors()
    assert mp.mean is None
    assert mp.cov is None
    assert mp.Lb is None
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=2.)
    assert_allclose(mp.mean, np.array([1., 2.]))
    assert_allclose(mp.cov, 2.)
    assert mp.Lb is None
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([2., 2.]))
    assert_allclose(mp.mean, np.array([1., 2.]))
    assert_allclose(mp.cov, np.array([2., 2.]))
    assert mp.Lb is None
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([[2., 2.], [2., 3.]]))
    assert_allclose(mp.mean, np.array([1., 2.]))
    assert_allclose(mp.cov, np.array([[2., 2.], [2., 3.]]))
    assert_allclose(mp.Lb[0], cho_factor(np.array([[2., 2.], [2., 3.]]))[0])
    
    with pytest.raises(ValueError):
        MeanPriors(mean=np.array([]))
        
    with pytest.raises(AssertionError):
        MeanPriors(mean=np.array([1., 2.]), cov=-1.)
        
    with pytest.raises(AssertionError):
        MeanPriors(mean=np.array([1., 2.]), cov=np.array([-1., 2.]))
        
    with pytest.raises(AssertionError):
        MeanPriors(mean=np.array([1., 2.]), cov=np.array([[-1., 2.], [2., 3.]]))
        
    with pytest.raises(AssertionError):
        MeanPriors(mean=np.array([1., 2.]), cov=np.ones(3))
    
    with pytest.raises(AssertionError):
        MeanPriors(mean=np.array([1., 2.]), cov=np.ones((2, 3)))
        
    with pytest.raises(ValueError):
        MeanPriors(mean=np.array([1., 2.]), cov=np.ones((3, 1, 1)))   

def test_MeanPriors_dm_dot_b():
    "test the function to take dot product of design matrix with prior mean"
    
    mp = MeanPriors()
    
    assert_allclose(mp.dm_dot_b(np.array([[1., 1.], [1., 2.]])), np.zeros(2))
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=2.)
    
    assert_allclose(mp.dm_dot_b(np.array([[1., 1.], [1., 2.]])), np.array([3., 5.]))

def test_MeanPriors_inv_cov():
    "test the routine to invert the covariance matrix in MeanPriors"
    
    mp = MeanPriors()
    assert_allclose(mp.inv_cov(), 0.)
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=2.)
    assert_allclose(mp.inv_cov(), np.eye(2)/2.)
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([2., 1.]))
    assert_allclose(mp.inv_cov(), np.array([[0.5, 0.], [0., 1.]]))
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([[2., 2.], [2., 3.]]))
    assert_allclose(np.dot(mp.inv_cov(), np.array([[2., 2.], [2., 3.]])), np.eye(2), atol=1.e-10)

def test_MeanPriors_inv_cov_b():
    "test the routine to invert the covariance matrix in MeanPriors times the mean"
    
    mp = MeanPriors()
    assert_allclose(mp.inv_cov_b(), 0.)
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=2.)
    assert_allclose(mp.inv_cov_b(), np.dot(np.eye(2)/2., np.array([1., 2.])))
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([2., 1.]))
    assert_allclose(mp.inv_cov_b(), np.dot(np.array([[0.5, 0.], [0., 1.]]), np.array([1., 2.])))
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([[2., 2.], [2., 3.]]))
    assert_allclose(mp.inv_cov_b(), np.linalg.solve(np.array([[2., 2.], [2., 3.]]), np.array([1., 2.])))
    
def test_MeanPriors_logdet_cov():
    "test the routine to invert the covariance matrix in MeanPriors times the mean"
    
    mp = MeanPriors()
    assert_allclose(mp.logdet_cov(), 0.)
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=2.)
    assert_allclose(mp.logdet_cov(), np.log(np.linalg.det(2.*np.eye(2))))
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([2., 1.]))
    assert_allclose(mp.logdet_cov(), np.log(np.linalg.det(np.array([[2., 0.], [0., 1.]]))))
    
    mp = MeanPriors(mean=np.array([1., 2.]), cov=np.array([[2., 2.], [2., 3.]]))
    assert_allclose(mp.logdet_cov(), np.log(np.linalg.det(np.array([[2., 2.], [2., 3.]]))))

def test_WeakPrior():
    "test the weakprior object"
    
    wp = WeakPrior()
    
    assert wp.logp(1.) == 0.
    assert wp.dlogpdtheta(1., CovTransform) == 0.
    assert wp.d2logpdtheta2(1., CorrTransform) == 0.
    assert wp.dlogpdx(1.) == 0.
    assert wp.d2logpdx2(1.) == 0.
    
    val = wp.sample()
    assert val >= -2.5
    assert val <= 2.5

@pytest.fixture
def dx():
    return 1.e-6

def test_Prior_transforms():
    "test transforms of prior distributions"
    
    class TestPrior(PriorDist):
        def logp(self, x):
            return 0.5*x**2
        def dlogpdx(self, x):
            return x
        def d2logpdx2(self, x):
            return 1.
        def sample_x(self):
            return 1.
            
    tp = TestPrior()
    assert_allclose(tp.logp(1.), 0.5)
    assert_allclose(tp.dlogpdx(1.), 1.)
    assert_allclose(tp.d2logpdx2(1.), 1.)
    assert_allclose(tp.sample_x(), 1.)
    
    assert_allclose(tp.dlogpdtheta(1., CovTransform), CovTransform.dscaled_draw(1.))
    assert_allclose(tp.d2logpdtheta2(1., CorrTransform),
                    CorrTransform.dscaled_draw(1.)**2 + CorrTransform.d2scaled_draw2(1.))
    assert_allclose(tp.sample(CovTransform), CovTransform.inv_transform(1.))
    

def test_NormalPrior(dx):
    "test the NormalPrior class"

    normprior = NormalPrior(2., 3.)

    assert_allclose(normprior.logp(0.5), np.log(norm.pdf(0.5, loc=2., scale=3.)))

    assert_allclose(normprior.dlogpdx(0.5),
                    (normprior.logp(0.5) - normprior.logp(0.5 - dx))/dx, atol=1.e-7, rtol=1.e-7)

    assert_allclose(normprior.d2logpdx2(0.5),
                    (normprior.dlogpdx(0.5) - normprior.dlogpdx(0.5 - dx))/dx, atol=1.e-7, rtol=1.e-7)
                    
    s = normprior.sample_x()
    
    with pytest.raises(AssertionError):
        NormalPrior(2., -1.)

def test_LogNormalPrior(dx):
    "test the LogNormalPrior class"

    lognormprior = LogNormalPrior(2., 3.)

    assert_allclose(lognormprior.logp(0.5), np.log(lognorm.pdf(0.5, 2., scale=3.)))

    assert_allclose(lognormprior.dlogpdx(0.5),
                    (lognormprior.logp(0.5) - lognormprior.logp(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

    assert_allclose(lognormprior.d2logpdx2(0.5),
                    (lognormprior.dlogpdx(0.5) - lognormprior.dlogpdx(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)
                    
    s = lognormprior.sample_x()
    assert s > 0.
                    
    with pytest.raises(AssertionError):
        LogNormalPrior(2., -1.)
    
    with pytest.raises(AssertionError):
        LogNormalPrior(-2., 1.)

def test_GammaPrior(dx):
    "test the GammaPrior class"

    gprior = GammaPrior(2., 3.)

    assert_allclose(gprior.logp(0.5), np.log(gamma.pdf(0.5, 2., scale=3.)))

    assert_allclose(gprior.dlogpdx(0.5),
                    (gprior.logp(0.5) - gprior.logp(0.5 - dx))/dx, atol=1.e-6, rtol=1.e-6)

    assert_allclose(gprior.d2logpdx2(0.5),
                    (gprior.dlogpdx(0.5) - gprior.dlogpdx(0.5 - dx))/dx, atol=1.e-5, rtol=1.e-6)
                    
    s = gprior.sample_x()
    assert s > 0.
                    
    with pytest.raises(AssertionError):
        GammaPrior(2., -1.)
    
    with pytest.raises(AssertionError):
        GammaPrior(-2., 1.)

def test_InvGammaPrior(dx):
    "test the InvGammaPrior class"

    igprior = InvGammaPrior(2., 3.)

    assert_allclose(igprior.logp(0.5), np.log(invgamma.pdf(0.5, 2., scale=3.)))

    assert_allclose(igprior.dlogpdx(0.5),
                    (igprior.logp(0.5) - igprior.logp(0.5 - dx))/dx, atol=1.e-5, rtol=1.e-5)

    assert_allclose(igprior.d2logpdx2(0.5),
                    (igprior.dlogpdx(0.5) - igprior.dlogpdx(0.5 - dx))/dx, atol=1.e-5, rtol=1.e-5)
                    
    s = igprior.sample_x()
    assert s > 0.
                    
    with pytest.raises(AssertionError):
        InvGammaPrior(2., -1.)
    
    with pytest.raises(AssertionError):
        InvGammaPrior(-2., 1.)
    
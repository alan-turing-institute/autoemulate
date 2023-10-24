import numpy as np
from numpy.testing import assert_allclose
import pytest
from inspect import signature
import types
from ..ExperimentalDesign import LatinHypercubeDesign
from ..SequentialDesign import SequentialDesign, MICEDesign, MICEFastGP
from ..GaussianProcess import GaussianProcess
from ..fitting import fit_GP_MAP
from tempfile import TemporaryFile
from ..LibGPGPU import gpu_usable

GPU_NOT_FOUND_MSG = "A compatible GPU could not be found or the GPU library (libgpgpu) could not be loaded"

def test_SequentialDesign_init():
    "test the init method of ExperimentalDesign"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed)
    assert type(sd.base_design).__name__ == 'LatinHypercubeDesign'
    assert sd.f == None
    assert sd.n_samples == None
    assert sd.n_init == 10
    assert sd.n_cand == 50
    assert sd.current_iteration == 0
    assert not sd.initialized
    assert sd.inputs == None
    assert sd.targets == None
    assert sd.candidates == None

    sd = SequentialDesign(ed, f, n_samples = 100, n_init = 20, n_cand = 100)
    assert type(sd.base_design).__name__ == 'LatinHypercubeDesign'
    assert callable(sd.f)
    assert len(signature(sd.f).parameters) == 1
    assert sd.n_samples == 100
    assert sd.n_init == 20
    assert sd.n_cand == 100
    assert sd.current_iteration == 0
    assert not sd.initialized
    assert sd.inputs == None
    assert sd.targets == None
    assert sd.candidates == None

def test_SequentialDesign_init_failures():
    "test cases where init should fail"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    with pytest.raises(TypeError):
        sd = SequentialDesign(3., f)

    with pytest.raises(TypeError):
        sd = SequentialDesign(ed, 3.)

    def f2(x, y):
        return np.array([1.])

    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f2)

    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, n_samples = -1)

    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, n_init = -1)

    with pytest.raises(ValueError):
        sd = SequentialDesign(ed, f, n_cand = -1)

def test_SequentialDesign_save_design():
    "test the save_design method"

    ed = LatinHypercubeDesign(3)
    sd = SequentialDesign(ed, n_init = 2, n_cand = 3)

    sd.inputs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    sd.targets = np.array([2., 4.])
    sd.candidates = np.array([[0.7, 0.8, 0.9], [0.15, 0.25, 0.35], [0.45, 0.55, 0.65]])

    with TemporaryFile() as tmp:
        sd.save_design(tmp)
        tmp.seek(0)
        design_file = np.load(tmp)
        assert_allclose(design_file['inputs'], np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
        assert_allclose(design_file['targets'], np.array([2., 4.]))
        assert_allclose(design_file['candidates'], np.array([[0.7, 0.8, 0.9],
                                                             [0.15, 0.25, 0.35], [0.45, 0.55, 0.65]]))

def test_SequentialDesign_load_design():
    "test the load_design method"

    ed = LatinHypercubeDesign(3)
    sd = SequentialDesign(ed, n_init = 2, n_cand = 3)

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                                      targets = np.array([2., 4.]),
                                      candidates = np.array([[0.7, 0.8, 0.9],
                                                             [0.15, 0.25, 0.35],
                                                             [0.45, 0.55, 0.65]]))
        tmp.seek(0)
        sd.load_design(tmp)

    assert_allclose(sd.inputs, np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    assert_allclose(sd.targets, np.array([2., 4.]))
    assert_allclose(sd.candidates, np.array([[0.7, 0.8, 0.9],
                                             [0.15, 0.25, 0.35],
                                             [0.45, 0.55, 0.65]]))
    assert sd.initialized
    assert sd.current_iteration == 2

    # disagreement between base design and inputs

    ed = LatinHypercubeDesign(2)
    sd = SequentialDesign(ed, n_init = 2, n_cand = 3)

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                                      targets = np.array([2., 4.]),
                                      candidates = np.array([[0.7, 0.8, 0.9],
                                                             [0.15, 0.25, 0.35],
                                                             [0.45, 0.55, 0.65]]))
        tmp.seek(0)
        with pytest.raises(AssertionError):
            sd.load_design(tmp)

    # disagreement between shape of inputs and candidates

    ed = LatinHypercubeDesign(3)
    sd = SequentialDesign(ed, n_init = 2, n_cand = 3)

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                                      targets = np.array([2., 4.]),
                                      candidates = np.array([[0.7, 0.8],
                                                             [0.15, 0.25],
                                                             [0.45, 0.55]]))
        tmp.seek(0)
        with pytest.raises(AssertionError):
            sd.load_design(tmp)

    # error due to targets having a bad shape

    ed = LatinHypercubeDesign(3)
    sd = SequentialDesign(ed, n_init = 2, n_cand = 3)

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                                      targets = np.array([2., 4., 5.]),
                                      candidates = np.array([[0.7, 0.8],
                                                             [0.15, 0.25],
                                                             [0.45, 0.55]]))
        tmp.seek(0)
        with pytest.raises(AssertionError):
            sd.load_design(tmp)

    # error due to existing targets but no inputs

    ed = LatinHypercubeDesign(3)
    sd = SequentialDesign(ed, n_init = 2, n_cand = 3)

    with TemporaryFile() as tmp:
        np.savez(tmp, inputs = None, targets = np.array([2., 4., 5.]), candidates = None)
        tmp.seek(0)
        with pytest.raises(AssertionError):
            sd.load_design(tmp)

def test_SequentialDesign_has_function():
    "test has_function method"

    ed = LatinHypercubeDesign(3)

    sd = SequentialDesign(ed)
    assert not sd.has_function()

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f)
    assert sd.has_function()

def test_SequentialDesign_get_n_parameters():
    "test the get_n_parameters method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f)
    assert sd.get_n_parameters() == 3

def test_SequentialDesign_get_n_init():
    "test the get_n_init method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f, n_init = 20)
    assert sd.get_n_init() == 20

def test_SequentialDesign_get_n_samples():
    "test the get_n_cand method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f, n_samples = 20)
    assert sd.get_n_samples() == 20

def test_SequentialDesign_get_n_cand():
    "test the get_n_cand method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f, n_cand = 20)
    assert sd.get_n_cand() == 20


def test_SequentialDesign_get_current_iteration():
    "test the get_current_iteration method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f)
    assert sd.get_current_iteration() == 0

    sd.current_iteration = 20
    assert sd.get_current_iteration() == 20

def test_SequentialDesign_get_inputs():
    "test the get_inputs method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f)
    assert sd.get_inputs() == None

    sd.inputs = np.zeros((3, 4))
    assert_allclose(sd.get_inputs(), np.zeros((3, 4)))

def test_SequentialDesign_get_targets():
    "test the get_targets method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f)
    assert sd.get_targets() == None

    sd.targets = np.zeros(4)
    assert_allclose(sd.get_targets(), np.zeros(4))

def test_SequentialDesign_get_candidates():
    "test the get_candidates method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f)
    assert sd.get_candidates() == None

    sd.candidates = np.zeros((3, 4))
    assert_allclose(sd.get_candidates(), np.zeros((3, 4)))

def test_SequentialDesign_get_base_design():
    "test the get_base_design method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f)
    assert sd.get_base_design() == "LatinHypercubeDesign"

def test_SequentialDesign_generate_initial_design():
    "test the generate_initial_design method"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.array([1.])

    sd = SequentialDesign(ed, f, n_init = 4)
    initial_design_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                        [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                        [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                        [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    initial_design_actual = sd.generate_initial_design()
    assert_allclose(initial_design_actual, initial_design_expected)
    assert_allclose(sd.inputs, initial_design_expected)
    assert sd.current_iteration == 4

    sd = SequentialDesign(ed, f, n_init = 4)
    sd.run_initial_design()

    with pytest.raises(AssertionError):
        sd.generate_initial_design()

def test_SequentialDesign_set_initial_targets():
    "test the set_initial_targets method"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    sd = SequentialDesign(ed, f, n_init = 4)
    initial_design_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                        [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                        [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                        [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    targets_expected = np.array([np.sum(i) for i in initial_design_expected])
    sd.generate_initial_design()
    sd.set_initial_targets(targets_expected)
    assert_allclose(sd.targets, np.reshape(targets_expected, (4,)))
    assert sd.initialized

    sd = SequentialDesign(ed, f, n_init = 4)
    targets_expected = np.array([np.sum(i) for i in initial_design_expected])
    sd.generate_initial_design()
    sd.set_initial_targets(targets_expected)
    assert_allclose(sd.targets, targets_expected)
    assert sd.initialized

    sd = SequentialDesign(ed, f, n_init = 4)
    with pytest.raises(ValueError):
        sd.set_initial_targets(targets_expected)

    sd = SequentialDesign(ed, f, n_init = 4)
    sd.generate_initial_design()
    with pytest.raises(AssertionError):
        sd.set_initial_targets(np.zeros((2, 4)))

    sd = SequentialDesign(ed, f, n_init = 4)
    sd.inputs = np.zeros((5,2))
    with pytest.raises(AssertionError):
        sd.set_initial_targets(targets_expected)

def test_SequentialDesign_run_initial_design():
    "test method to run initial design"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    sd = SequentialDesign(ed, f, n_init = 4)
    initial_design_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                        [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                        [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                        [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    targets_expected = np.array([np.sum(i) for i in initial_design_expected])
    sd.run_initial_design()
    assert_allclose(sd.inputs, initial_design_expected)
    assert_allclose(sd.targets, targets_expected)
    assert sd.initialized
    assert sd.current_iteration == 4

    sd = SequentialDesign(ed)
    with pytest.raises(AssertionError):
        sd.run_initial_design()

def test_SequentialDesign_generate_candidates():
    "test the _generate_candidates method"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    sd = SequentialDesign(ed, f, n_cand = 4)
    candidates_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                    [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                    [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                    [0.4531112960399023, 0.3977273628763245, 0.5867585643640021]])
    sd._generate_candidates()
    assert_allclose(sd.candidates, candidates_expected)

def test_SequentialDesign_eval_metric():
    "test the _eval_metric method"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    sd = SequentialDesign(ed, f)

    with pytest.raises(NotImplementedError):
        sd._eval_metric()

def test_SequentialDesign_estimate_next_target():
    "test the estimate next point method of sequential design"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    sd = SequentialDesign(ed, f)

    with pytest.raises(NotImplementedError):
        sd._estimate_next_target(np.array([0., 1., 2.]))

def test_SequentialDesign_get_batch_points():
    "test the get_batch_points method of a MICE design"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    def tmp_eval_metric(self):
        return 0

    def tmp_estimate_next_target(self, next_point):
        return np.array([1.])

    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd._estimate_next_target = types.MethodType(tmp_estimate_next_target, sd)

    sd.run_initial_design()

    next_points = sd.get_batch_points(2)

    assert_allclose(next_points, np.array([[3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01],
                                           [0.1314127131166828, 0.3850568631590907, 0.2234836206262954]]))
    assert_allclose(sd.inputs, np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                         [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                         [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                         [0.4531112960399023, 0.3977273628763245, 0.5867585643640021],
                                         [3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01],
                                         [0.1314127131166828, 0.3850568631590907, 0.2234836206262954]]))
    assert_allclose(sd.targets, np.array([1.4316938134430974, 1.1421640156963757, 2.1823571108117488,
                                          1.437597223280229 ]))
    assert sd.current_iteration == 4

    # check raises error if number of batch points is negative

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    sd.run_initial_design()

    with pytest.raises(AssertionError):
        sd.get_batch_points(-1)

def test_SequentialDesign_get_next_point():
    "test the get_next_point method"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    def tmp_eval_metric(self):
        return 0

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()
    next_point = sd.get_next_point()

    next_point_expected = np.array([3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01])
    inputs_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                [0.4531112960399023, 0.3977273628763245, 0.5867585643640021],
                                [3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01]])
    candidates_expected = np.array([[3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01],
                                    [6.7353171430888004e-01, 3.2967090024809237e-01, 4.7561098969781079e-01],
                                    [9.9391552790043791e-01, 5.0015522522360367e-01, 2.4850731127782361e-01],
                                    [6.6098271480205528e-04, 9.7616245971529891e-01, 7.4815518971467765e-01]])
    assert_allclose(next_point, next_point_expected)
    assert_allclose(sd.inputs, inputs_expected)
    assert_allclose(sd.candidates, candidates_expected)


    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    sd.generate_initial_design()
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)

    with pytest.raises(ValueError):
        next_point = sd.get_next_point()

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    with pytest.raises(ValueError):
        next_point = sd.get_next_point()

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()
    sd.inputs = np.zeros((5,3))

    with pytest.raises(AssertionError):
        next_point = sd.get_next_point()

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()
    sd.targets = np.zeros((5,3))

    with pytest.raises(AssertionError):
        next_point = sd.get_next_point()

def test_SequentialDesign_set_batch_targets():
    "test the set_batch_targets method"

    ed = LatinHypercubeDesign(3)

    sd = SequentialDesign(ed, n_init = 4, n_cand = 4)

    sd.inputs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
                          [0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]])
    sd.targets = np.array([1., 2., 3., 4.])
    sd.current_iteration = 4
    sd.initialized = True

    sd.set_batch_targets(np.array([5., 6.]))

    assert_allclose(sd.targets, np.array([1., 2., 3., 4., 5., 6.]))
    assert sd.current_iteration == 6

    # failure if new_targets wrong shape

    ed = LatinHypercubeDesign(3)

    sd = SequentialDesign(ed, n_init = 4, n_cand = 4)

    sd.inputs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
                          [0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]])
    sd.targets = np.array([1., 2., 3., 4.])

    with pytest.raises(AssertionError):
        sd.set_batch_targets(np.array([5., 6., 7.]))

def test_SequentialDesign_set_next_target():
    "test the set_next_target method"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    def tmp_eval_metric(self):
        return 0

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()
    next_point = sd.get_next_point()
    new_target = np.reshape(np.array(np.sum(next_point)), (1,))

    inputs_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                [0.4531112960399023, 0.3977273628763245, 0.5867585643640021],
                                [3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01]])
    targets_expected = np.array([np.sum(i) for i in inputs_expected])

    sd.set_next_target(new_target)
    assert_allclose(sd.targets, targets_expected)
    assert sd.current_iteration == 5


    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()
    next_point = sd.get_next_point()

    with pytest.raises(AssertionError):
        sd.set_next_target(np.zeros((2,)))


    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    sd.generate_initial_design()
    sd.inputs = np.zeros((5,3))
    next_target = np.zeros(1)

    with pytest.raises(ValueError):
        sd.set_next_target(next_target)

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    with pytest.raises(ValueError):
        sd.set_next_target(next_target)

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()

    with pytest.raises(AssertionError):
        sd.set_next_target(next_target)

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()
    sd.get_next_point()
    sd.targets = np.zeros((5,3))

    with pytest.raises(AssertionError):
        sd.set_next_target(next_target)

def test_SequentialDesign_run_next_point():
    "test the run_next_point method"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    def tmp_eval_metric(self):
        return 0

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_initial_design()
    sd.run_next_point()

    inputs_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                [0.4531112960399023, 0.3977273628763245, 0.5867585643640021],
                                [3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01]])
    targets_expected = np.array([np.sum(i) for i in inputs_expected])

    assert_allclose(sd.inputs, inputs_expected)
    assert_allclose(sd.targets, targets_expected)
    assert sd.current_iteration == 5

    sd = SequentialDesign(ed)
    with pytest.raises(AssertionError):
        sd.run_next_point()


def test_SequentialDesign_run_sequential_design():
    "test the run_sequential_design method"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    def tmp_eval_metric(self):
        return 0

    inputs_expected = np.array([[0.9660431763890672, 0.2080126306969736, 0.2576380063570568],
                                [0.0684779421445063, 0.9308367720360009, 0.1428493015158686],
                                [0.6345029195085983, 0.6651343562344474, 0.8827198350687029],
                                [0.4531112960399023, 0.3977273628763245, 0.5867585643640021],
                                [3.9602716910300234e-01, 4.3469440375712098e-02, 9.3294684823072194e-01],
                                [0.1314127131166828, 0.3850568631590907, 0.2234836206262954],
                                [0.1648353557812244, 0.0994384529732742, 0.4715221513612055],
                                [0.8739475357732106, 0.058541390000348 , 0.3103313392459021]])
    targets_expected = np.array([np.sum(i) for i in inputs_expected])

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_sequential_design(4)

    assert_allclose(sd.inputs, inputs_expected)
    assert_allclose(sd.targets, targets_expected)
    assert sd.current_iteration == 8

    np.random.seed(74632)

    sd = SequentialDesign(ed, f, n_samples = 4, n_init = 4, n_cand = 4)
    sd._eval_metric = types.MethodType(tmp_eval_metric, sd)
    sd.run_sequential_design()

    assert_allclose(sd.inputs, inputs_expected)
    assert_allclose(sd.targets, targets_expected)
    assert sd.current_iteration == 8

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)

    with pytest.raises(ValueError):
        sd.run_sequential_design()

    sd = SequentialDesign(ed)
    with pytest.raises(AssertionError):
        sd.run_sequential_design()

    sd = SequentialDesign(ed, f, n_init = 4, n_cand = 4)
    with pytest.raises(AssertionError):
        sd.run_sequential_design(n_samples = -1)

def test_SequentialDesign_str():
    "test string method of sequential design"

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    expected_string = ""
    expected_string += "SequentialDesign with\n"
    expected_string += "LatinHypercubeDesign base design\n"
    expected_string += "None total samples\n"
    expected_string += "10 initial points\n"
    expected_string += "50 candidate points\n"
    expected_string += "0 current samples\n"
    expected_string += "current inputs: None\n"
    expected_string += "current targets: None"

    sd = SequentialDesign(ed)
    assert str(sd) == expected_string

    expected_string = ""
    expected_string += "SequentialDesign with\n"
    expected_string += "LatinHypercubeDesign base design\n"
    expected_string += "a bound simulator function\n"
    expected_string += "10 total samples\n"
    expected_string += "5 initial points\n"
    expected_string += "10 candidate points\n"
    expected_string += "0 current samples\n"
    expected_string += "current inputs: None\n"
    expected_string += "current targets: None"

    sd = SequentialDesign(ed, f, n_samples = 10, n_init = 5, n_cand = 10)
    assert str(sd) == expected_string

def test_MICEDesign_init():
    "test the init method of MICEDesign"

    ed = LatinHypercubeDesign(3)

    md = MICEDesign(ed)

    assert type(md.base_design).__name__ == 'LatinHypercubeDesign'
    assert md.f == None
    assert md.n_samples == None
    assert md.n_init == 10
    assert md.n_cand == 50
    assert md.current_iteration == 0
    assert not md.initialized
    assert md.inputs == None
    assert md.targets == None
    assert md.candidates == None
    assert md.nugget == "adaptive"
    assert_allclose(md.nugget_s, 1.)

    def f(x):
        return np.array([1.])

    md = MICEDesign(ed, f, 20, 5, 40, 1.e-12, 0.1)

    assert type(md.base_design).__name__ == 'LatinHypercubeDesign'
    assert callable(md.f)
    assert len(signature(md.f).parameters) == 1
    assert md.n_samples == 20
    assert md.n_init == 5
    assert md.n_cand == 40
    assert md.current_iteration == 0
    assert not md.initialized
    assert md.inputs == None
    assert md.targets == None
    assert md.candidates == None
    assert_allclose(md.nugget, 1.e-12)
    assert_allclose(md.nugget_s, 0.1)

def test_MICEDesign_init_failures():
    "test occasions where MICE Design should fail upon initializing"

    ed = LatinHypercubeDesign(3)

    with pytest.raises(ValueError):
        md = MICEDesign(ed, nugget = -1.)

    with pytest.raises(ValueError):
        md = MICEDesign(ed, nugget_s = -1.)

def test_MICEDesign_get_nugget():
    "test the get_nugget method of MICE Design"

    ed = LatinHypercubeDesign(3)

    md = MICEDesign(ed)

    assert md.get_nugget() == "adaptive"

    md = MICEDesign(ed, nugget = 1.)

    assert_allclose(md.get_nugget(), 1.)

def test_MICEDesign_get_nugget_s():
    "test the get_nugget_s method of MICE Design"

    ed = LatinHypercubeDesign(3)

    md = MICEDesign(ed)

    assert_allclose(md.get_nugget_s(), 1.)

## NOTE: The MICE test that follows does not work as the GP class has evolved significantly
## since the MICE code base was written. This will be fixed once the new interface to
## the GP class is clarified and implemented. However, no changes have been made
## that affect the correctness of the results, just the test cases were worked
## out based on an old default version that no longer applies.

@pytest.mark.skip
def test_MICEDesign_estimate_next_target():
    "test the estimate next target method for a MICE design"

    ed = LatinHypercubeDesign(3)
    md = MICEDesign(ed)

    md.gp = GaussianProcess(np.array([[1., 2., 3.], [4., 5., 6]]), np.array([2., 4.]))
    md.gp.theta = np.zeros(md.gp.n_params)

    assert_allclose(np.array([0.0018237589305011]), md._estimate_next_target(np.zeros(3)))

    with pytest.raises(AssertionError):
        md._estimate_next_target(np.zeros(2))

@pytest.mark.skip
def test_MICEDesign_MICE_criterion():
    "test the function to compute the MICE criterion"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    md = MICEDesign(ed, f, n_init = 4, n_cand = 4)

    md.run_initial_design()
    md._generate_candidates()

    md.gp = GaussianProcess(md.get_inputs(), md.get_targets())
    md.gp = fit_GP_MAP(md.gp)

    md.gp_fast = MICEFastGP(md.candidates, np.ones(4), nugget=1.)
    md.gp_fast.theta = md.gp.theta

    metric = md._MICE_criterion(0)

    metric_expected = 0.0899338596342571

    assert_allclose(metric, metric_expected, rtol = 1.e-6)

    with pytest.raises(AssertionError):
        metric = md._MICE_criterion(-1)

    with pytest.raises(AssertionError):
        metric = md._MICE_criterion(5)

@pytest.mark.skip
def test_MICEDesign_eval_metric():
    "test the _eval_metric method of MICE Design"

    np.random.seed(74632)

    ed = LatinHypercubeDesign(3)

    def f(x):
        return np.sum(x)

    md = MICEDesign(ed, f, n_init = 4, n_cand = 4)

    md.run_initial_design()
    md._generate_candidates()

    best_point = md._eval_metric()

    best_point_expected = 0

    assert best_point == best_point_expected

@pytest.mark.skip
def test_MICEFastGP():
    "test the correction formula for the modified GP for Fast MICE"

    gp = MICEFastGP(np.reshape([1., 2., 3., 4], (4, 1)), [1., 1., 1., 1.])
    gp.theta = np.array([0., -1.])
    result = gp.fast_predict(3)
    result_expected = 1.191061906777163

    assert_allclose(result, result_expected)

    with pytest.raises(AssertionError):
        result = gp.fast_predict(-1)

    with pytest.raises(AssertionError):
        result = gp.fast_predict(5)

import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..formula import mean_from_string, _convert_token, _is_float, _token_to_mean
from ..formula import _tokenize_string, _parse_tokens, _eval_parsed_tokens
from ..MeanFunction import ConstantMean, Coefficient, LinearMean
from ..MeanFunction import MeanSum, MeanProduct, MeanPower, MeanComposite
try:
    import patsy
    no_patsy = False
    from ..formula import mean_from_patsy_formula, _term_to_mean
except ImportError:
    no_patsy = True

patsy_skip = pytest.mark.skipif(no_patsy, reason="patsy is needed to test formula parsing")

@patsy_skip
@pytest.mark.parametrize("code,inputdict,params,resulttype,result",
                         [("x[0] - 1",
                           {}     , np.ones(1)                , MeanProduct , np.array([1., 4.])),
                          ("a - 1",
                           {}     , np.ones(2)                , MeanProduct , np.array([1., 1.])),
                          ("a - 1",
                           {"a":1}, np.ones(1)                , MeanProduct , np.array([2., 5.])),
                          ("x[0]",
                           {}     , np.array([1., 2.])        , MeanSum    , np.array([3., 9.])),
                          ("x[1] - 1",
                           {}     , np.array([2.])            , MeanProduct, np.array([4., 10.])),
                          ("I ( inputs[0]^2) - 1",
                           {}     , np.ones(1)                , MeanProduct, np.array([1., 16.])),
                          ("x[0]*x[1]",
                           {}     , np.array([1., 2., 3., 4.]), MeanSum    , np.array([17., 104.])),
                          (patsy.ModelDesc.from_formula("x[0] - 1"),
                           {}     , np.ones(1)                , MeanProduct , np.array([1., 4.])),
                          (patsy.ModelDesc.from_formula("a - 1"),
                           {}     , np.ones(2)                , MeanProduct , np.array([1., 1.])),
                          (patsy.ModelDesc.from_formula("a - 1"),
                           {"a":1}, np.ones(1)                , MeanProduct , np.array([2., 5.])),
                          (patsy.ModelDesc.from_formula("x[0]"),
                           {}     , np.array([1., 2.])        , MeanSum    , np.array([3., 9.])),
                          (patsy.ModelDesc.from_formula("x[1] - 1"),
                           {}     , np.array([2.])            , MeanProduct, np.array([4., 10.])),
                          (patsy.ModelDesc.from_formula("I ( inputs[0]^2) - 1"),
                           {}     , np.ones(1)                , MeanProduct, np.array([1., 16.])),
                          (patsy.ModelDesc.from_formula("x[0]*x[1]"),
                           {}     , np.array([1., 2., 3., 4.]), MeanSum    , np.array([17., 104.]))])
def test_mean_from_patsy_formula(code, inputdict, params, resulttype, result):
    "test the mean_from_patsy_formula function"

    x = np.array([[1., 2., 3.], [4., 5., 6.]])

    mf = mean_from_patsy_formula(code, inputdict)

    assert isinstance(mf, resulttype)
    assert mf.get_n_params(x) == len(params)

    assert_allclose(mf.mean_f(x, params), result)

@pytest.mark.parametrize("code,inputdict,params,resulttype,result",
                         [("x[0]",
                           {}     , np.zeros(0)       , LinearMean , np.array([1., 4.])),
                          ("y = x[0]",
                           {}     , np.zeros(0)       , LinearMean , np.array([1., 4.])),
                          ("y ~ x[0]",
                           {}     , np.zeros(0)       , LinearMean , np.array([1., 4.])),
                          ("~ x[0]",
                           {}     , np.zeros(0)       , LinearMean , np.array([1., 4.])),
                          ("= x[0]",
                           {}     , np.zeros(0)       , LinearMean , np.array([1., 4.])),
                          ("a",
                           {}     , np.ones(1)        , Coefficient, np.array([1., 1.])),
                          ("a",
                           {"a":1}, np.zeros(0)       , LinearMean , np.array([2., 5.])),
                          ("a + b*x[0]",
                           {}     , np.array([1., 2.]), MeanSum    , np.array([3., 9.])),
                          ("a*x[1]",
                           {}     , np.array([2.])    , MeanProduct, np.array([4., 10.])),
                          ("inputs[0]**2",
                           {}     , np.zeros(0)       , MeanPower  , np.array([1., 16.])),
                          ("I ( inputs[0]^2)",
                           {}     , np.zeros(0)       , MeanPower  , np.array([1., 16.])),
                          ("(a + b*x[0])(x[0]*x[1])",
                           {}     , np.array([1., 2.]), MeanComposite, np.array([5., 41.]))])
def test_mean_from_string(code, inputdict, params, resulttype, result):
    "test the mean_from_string function"

    x = np.array([[1., 2., 3.], [4., 5., 6.]])

    mf = mean_from_string(code, inputdict)

    assert isinstance(mf, resulttype)
    assert mf.get_n_params(x) == len(params)

    assert_allclose(mf.mean_f(x, params), result)

def test_mean_from_string_failures():
    "test that mean_from_string correctly fails"

    with pytest.raises(AssertionError):
        mean_from_string(1)

@patsy_skip
def test_term_to_mean():
    "test the _term_to_mean function"

    t = patsy.ModelDesc.from_formula("x[0]").rhs_termlist

    x = np.array([[1., 2., 3.], [4., 5., 6.]])

    params = np.array([2.])

    type_expected = [ Coefficient, MeanProduct ]
    mean_expected = [ np.array([2., 2.]), np.array([2., 8.]) ]

    for term, type_exp, mean_exp in zip(t, type_expected, mean_expected):
        mf = _term_to_mean(term)
        assert mf.get_n_params(x) == len(params)
        assert isinstance(mf, type_exp)
        assert_allclose(mf.mean_f(x, params), mean_exp)

@patsy_skip
def test_term_to_mean_failures():
    "test situations where term_to_mean should fail"

    with pytest.raises(AssertionError):
        _term_to_mean(1)

    t = patsy.ModelDesc.from_formula("x").rhs_termlist[0]
    t.factors = [1, 2]

    with pytest.raises(AssertionError):
        _term_to_mean(t)

@pytest.mark.parametrize("token,inputdict,result", [("x[1]", {}, "x[1]"),
                                                   ("inputs[0]", {}, "x[0]"),
                                                   ("a", {}, "a"),
                                                   ("a", {"a":0}, "x[0]"),
                                                   ("d", {"d":1}, "x[1]")])
def test_convert_token(token, inputdict, result):
    "test the function to convert tokens"

    assert _convert_token(token, inputdict) == result

def test_convert_token_failures():
    "test situations where _convert_token should fail"

    with pytest.raises(AssertionError):
        _convert_token(1)

def test_is_float():
    "test the _is_float function"

    assert _is_float("1")
    assert _is_float("-2.e-4")
    assert not _is_float("a")

@pytest.mark.parametrize("token,inputdict,params,resulttype,result",
                         [("2."  , {}     , np.zeros(0), ConstantMean, np.array([2., 2.])),
                          ("a"   , {}     , np.ones(1) , Coefficient , np.array([1., 1.])),
                          ("x[0]", {}     , np.zeros(0), LinearMean  , np.array([1., 4.])),
                          ("a"   , {"a":0}, np.zeros(0), LinearMean  , np.array([1., 4.])),
                          ("x[1]", {}     , np.zeros(0), LinearMean  , np.array([2., 5.]))])
def test_token_to_mean(token, inputdict, params, resulttype, result):
    "test the _token_to_mean function"

    x = np.array([[1., 2., 3.], [4., 5., 6.]])

    mf = _token_to_mean(token, inputdict)

    assert isinstance(mf, resulttype)
    assert mf.get_n_params(x) == len(params)

    assert_allclose(mf.mean_f(x, params), result)

def test_token_to_mean_failures():
    "test situation where token_to_mean should fail"

    with pytest.raises(AssertionError):
        _token_to_mean(1)

    with pytest.raises(ValueError):
        _token_to_mean("x[0")

    with pytest.raises(ValueError):
        _token_to_mean("x(1]")

    with pytest.raises(ValueError):
        _token_to_mean("x[a]")

    with pytest.raises(AssertionError):
        _token_to_mean("x[-19]")

@pytest.mark.parametrize("code,result", [("a", ["a"]),
                                         ("y = a", ["a"]),
                                         ("y ~ a", ["a"]),
                                         ("= a", ["a"]),
                                         ("~ a", ["a"]),
                                         ("a + b", ["a", "+", "b"]),
                                         ("a*b+ c", ["a", "*", "b", "+", "c"]),
                                         ("x[0]", ["x[0]"]),
                                         ("x[ 0 ]", ["x[0]"]),
                                         ("x**2", ["x", "^", "2"]),
                                         ("x^2", ["x", "^", "2"]),
                                         ("c(d)", ["c", "(", "d", ")"]),
                                         ("a*(b + c)**3 + x[a]", ["a", "*", "(", "b", "+", "c", ")",
                                                                  "^", "3", "+", "x[a]"])])
def test_tokenize_string(code, result):
    "test the _tokenize_string function"

    assert _tokenize_string(code) == result

def test_tokenize_string_failures():
    "test situations where _tokenize_string should fail"

    with pytest.raises(AssertionError):
        _tokenize_string(1)

    with pytest.raises(SyntaxError):
        _tokenize_string("x[")

    with pytest.raises(SyntaxError):
        _tokenize_string("x]")

    with pytest.raises(SyntaxError):
        _tokenize_string("x[1] + x]")

    with pytest.raises(SyntaxError):
        _tokenize_string("x[(0)]")

    with pytest.raises(SyntaxError):
        _tokenize_string("call")

    with pytest.raises(SyntaxError):
        _tokenize_string("a + b = c")

    with pytest.raises(SyntaxError):
        _tokenize_string("a + b ~ c")

@pytest.mark.parametrize("token_list,output_list",
                         [(["a"], ["a"]),
                          (["a", "+", "b"], ["a", "b", "+"]),
                          (["a", "*", "b", "+", "c"], ["a", "b", "*", "c", "+"]),
                          (["x", "^", "2"], ["x", "2", "^"]),
                          (["c", "(", "d", ")"], ["c", "d", "call"]),
                          (["a", "*", "(", "b", "+", "c", ")", "^", "3", "+", "x[0]"],
                           ["a", "b", "c", "+", "3", "^", "*", "x[0]", "+"])])
def test_parse_tokens(token_list, output_list):
    "test the parse_tokens function"

    assert _parse_tokens(token_list) == output_list

def test_parse_tokens_failures():
    "test situation where parse_tokens fails"

    with pytest.raises(AssertionError):
        _parse_tokens("a")

    with pytest.raises(AssertionError):
        _parse_tokens([1])

    with pytest.raises(SyntaxError):
        _parse_tokens(["x" "+", "(", "a", "+", "b"])

    with pytest.raises(SyntaxError):
        _parse_tokens(["x" "*", "a", "+", "b", ")"])

    with pytest.raises(SyntaxError):
        _parse_tokens(["="])

    with pytest.raises(SyntaxError):
        _parse_tokens(["~"])

@pytest.mark.parametrize("stack,inputdict,params,resulttype,result",
                         [(["x[0]"],
                           {}     , np.zeros(0)       , LinearMean , np.array([1., 4.])),
                          (["a"],
                           {}     , np.ones(1),        Coefficient , np.array([1., 1.])),
                          (["a"],
                           {"a":1}, np.zeros(0)       , LinearMean , np.array([2., 5.])),
                          (["a", "b", "x[0]", "*", "+"],
                           {}     , np.array([1., 2.]), MeanSum    , np.array([3., 9.])),
                          (["a", "x[1]", "*"],
                           {}     , np.array([2.])    , MeanProduct, np.array([4., 10.])),
                          (["inputs[0]", "2", "^"],
                           {}     , np.zeros(0)       , MeanPower  , np.array([1., 16.])),
                          (["I", "inputs[0]", "2", "^", "call"],
                           {}     , np.zeros(0)       , MeanPower  , np.array([1., 16.])),
                          (["a", "b", "x[0]", "*", "+", "x[0]", "x[1]", "*", "call"],
                           {}     , np.array([1., 2.]), MeanComposite, np.array([5., 41.]))])
def test_eval_parsed_tokens(stack, inputdict, params, resulttype, result):
    "test the eval_parsed_tokens function"

    x = np.array([[1., 2., 3.], [4., 5., 6.]])

    mf = _eval_parsed_tokens(stack, inputdict)

    assert isinstance(mf, resulttype)
    assert mf.get_n_params(x) == len(params)

    assert_allclose(mf.mean_f(x, params), result)

def test_eval_parsed_tokens_failures():
    "test situations where eval_parsed_tokens should fail"

    with pytest.raises(AssertionError):
        _eval_parsed_tokens(1)

    with pytest.raises(AssertionError):
        _eval_parsed_tokens([1])

    with pytest.raises(SyntaxError):
        _eval_parsed_tokens(["a", "+"])

    with pytest.raises(SyntaxError):
        _eval_parsed_tokens(["a", "I", "+"])

    with pytest.raises(SyntaxError):
        _eval_parsed_tokens(["I", "a", "+"])

    with pytest.raises(SyntaxError):
        _eval_parsed_tokens(["a", "b"])

    with pytest.raises(SyntaxError):
        _eval_parsed_tokens(["="])

    with pytest.raises(SyntaxError):
        _eval_parsed_tokens(["~"])

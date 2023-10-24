from mogp_emulator import MeanFunction
try:
    from patsy import ModelDesc, Term, EvalFactor
    no_patsy = False
except ImportError:
    no_patsy = True

def mean_from_patsy_formula(formula, inputdict={}):
    """
    Create a mean function from a patsy formula

    This is the functional interface to creating a mean function from a patsy/R formula.
    The higher level function ``MeanFunction`` in the ``MeanFunction`` module is preferred
    over this one, but this potentially gives the user slightly more control as a
    patsy ``ModelDesc`` object can be passed directly, rather than giving a string.

    This method takes a string or a patsy ``ModelDesc`` object as an input and an optional
    dictionary that map strings to integer indices in the input data. The formula is then
    parsed with patsy, and the individual terms resulting from that are converted to
    mean functions and composed using the provided operations.

    The string formulas can be specified in several ways. The formula LHS is implicitly
    always ``"y = "`` or ``"y ~ "``, though these can be explicitly provided as well
    (though it is ignored in the conversion). The RHS may contain a set of terms
    containing the add, multiply, power, and call operations much in the same way
    that the operations would be entered as regular python code. Parentheses are
    used to indicated prececence as well as the call operation, and square brackets
    indicate an indexing operation on the inputs. Inputs may be specified as either
    a string such as ``"x[0]"``, ``"inputs[0]"``, or a string that can be mapped to
    an integer index with the optional dictionary passed to the function. Any strings
    not representing operations or inputs as described above are interpreted as follows:
    if the string can be converted into a number, then it is interpreted as a
    ``ConstantMean`` fixed mean function object; otherwise it is assumed to represent
    a fitting coefficient. Note that this means many characters that do not represent
    operations within this mean function language but would not normally be considered
    as python variables will nonetheless be converted into fitting coefficients --
    it is up to the user to get this right.

    Expressions that are repeated or redundant will not be simplified beyond the parsing
    done by patsy, so the user should take care that the provided expression is sensible
    as a mean function and will not cause problems when fitting.

    Examples: ::

        >>> from mogp_emulator.formula import mean_from_patsy_formula
        >>> mf1 = mean_from_patsy_formula("x[0]")
        >>> print(mf1)
        c + c*x[0]
        >>> mf2 = mean_from_patsy_formula("a*b", {"a": 0, "b": 1})
        >>> print(mf2)
        c + c*x[0] + c*x[1] + c*x[0]*x[1]

    :param formula: string representing the desired mean function formula
                    or a patsy ``ModelDesc`` object
    :type formula: str or ModelDesc
    :param inputdict: dictionary used to map variables to input indices. Maps
                      strings to integer indices (must be non-negative). Optional,
                      default is ``{}``.
    :type inputdict: dict
    :returns: New subclass of ``MeanBase`` implementing the given formula
    :rtype: subclass of MeanBase (exact type will depend on the formula that is provided)
    """

    assert not no_patsy, "patsy must be installed to parse formulas using patsy"

    if isinstance(formula, str):
        model = ModelDesc.from_formula(formula)
    elif isinstance(formula, ModelDesc):
        model = formula

    model_terms = []

    for term in model.rhs_termlist:
        model_terms.append(_term_to_mean(term, inputdict))

    mf = model_terms.pop(0)

    assert issubclass(type(mf), MeanFunction.MeanBase)

    for term in model_terms:
        mf += term

    assert issubclass(type(mf), MeanFunction.MeanBase)

    return mf

def mean_from_string(formula, inputdict={}):
    """
    Create a mean function from a string formula

    This is the functional interface to creating a mean function from a string formula.
    The higher level function ``MeanFunction`` in the ``MeanFunction`` module is preferred
    over this one, but this can also be used and the output is identical.

    This method takes a string as an input and an optional dictionary that map strings to
    integer indices in the input data.

    The string formulas can be specified in several ways. The formula LHS is implicitly
    always ``"y = "`` or ``"y ~ "``, though these can be explicitly provided as well.
    The RHS may contain a set of terms containing the add, multiply, power, and
    call operations much in the same way that the operations would be entered as
    regular python code. Parentheses are used to indicated prececence as well as
    the call operation, and square brackets indicate an indexing operation on the
    inputs. Inputs may be specified as either a string such as ``"x[0]"``,
    ``"inputs[0]"``, or a string that can be mapped to an integer index with the
    optional dictionary passed to the function. Any strings not representing operations
    or inputs as described above are interpreted as follows: if the string can
    be converted into a number, then it is interpreted as a ``ConstantMean`` fixed
    mean function object; otherwise it is assumed to represent a fitting coefficient.
    Note that this means many characters that do not represent operations within this
    mean function language but would not normally be considered as python variables
    will nonetheless be converted into fitting coefficients -- it is up to the user
    to get this right.

    Expressions that are repeated or redundant will not be simplified, so the user should
    take care that the provided expression is sensible as a mean function and will not
    cause problems when fitting.

    Examples: ::

        >>> from mogp_emulator.formula import mean_from_string
        >>> mf1 = mean_from_string("y = a + b*x[0]")
        >>> print(mf1)
        c + c*x[0]
        >>> mf2 = mean_from_string("c*a*b", {"a": 0, "b": 1})
        >>> print(mf2)
        c*x[0]*x[1]

    :param formula: string representing the desired mean function formula
    :type formula: str
    :param inputdict: dictionary used to map variables to input indices. Maps
                      strings to integer indices (must be non-negative). Optional,
                      default is ``{}``.
    :type inputdict: dict
    :returns: New subclass of ``MeanBase`` implementing the given formula
    :rtype: subclass of MeanBase (exact type will depend on the formula that is provided)
    """

    assert isinstance(formula, str)

    tokens = _tokenize_string(formula)
    eval_stack = _parse_tokens(tokens)

    mf = _eval_parsed_tokens(eval_stack, inputdict)

    assert issubclass(type(mf), MeanFunction.MeanBase)

    return mf

def _term_to_mean(term, inputdict={}):
    """
    Convert a patsy Term object into a mean function object

    Converts an individual term to a subclass of ``MeanBase`` appropriate for the
    formula (either will be a ``Coefficient`` or a ``MeanProduct`` instance).
    An implicit ``Coefficient`` is included, and then the results from parsing
    each of the factors in the term are multiplied by the coefficient. Variable
    substutions are done using the provided ``inputdict``, mapping strings
    to indices in the inputs, if there is one. Returns the mean function object
    appropriate for this term.

    :param term: patsy ``Term`` object
    :type formula: Term
    :param inputdict: dictionary used to map variables to input indices. Maps
                      strings to integer indices (must be non-negative). Optional,
                      default is ``{}``.
    :type inputdict: dict
    :returns: New subclass of ``MeanBase`` implementing the given term
    :rtype: subclass of MeanBase (exact type will depend on the formula that is provided)
    """

    assert not no_patsy, "patsy must be installed to parse formulas using patsy"

    assert isinstance(term, Term)

    # add leading coefficient and multiply by all factors

    mf = MeanFunction.Coefficient()

    for factor in term.factors:
        assert isinstance(factor, EvalFactor)
        mf *= mean_from_string(factor.code, inputdict)

    assert issubclass(type(mf), MeanFunction.MeanBase)

    return mf

def _convert_token(token, inputdict={}):
    """
    Converts an individual token into inputs

    Converts an individual token from the tokenizer to a dependent variable,
    making replacements if the given string has aliases. These include any
    strings found in the provided dictionary and converts ``'inputs'`` to
    ``'x'`` as this is an implicit alias.

    :param token: string token to be converted
    :type str:
    :param inputdict: dictionary used to map variables to input indices. Maps
                      strings to integer indices (must be non-negative). Optional,
                      default is ``{}``.
    :type inputdict: dict
    :returns: converted string token if any substitutions were found, otherwise
              the same token is returned
    :rtype: str
    """

    assert isinstance(token, str), "formula input to mean function is not a string"

    if token[:6] == "inputs":
        newtoken = "x"+token[6:]
    elif token in inputdict:
        newtoken = "x[{}]".format(inputdict[token])
    else:
        newtoken = token

    return newtoken

def _is_float(val):
    """
    Checks if a token can be converted into a float

    Returns a boolean indicating if the given token can be converted to a float

    :param val: input to be tested, meant to be a string but can be any type
    :type val: string (though any other type is also acceptable)
    :returns: boolean indicating if input can be converted to a float
    :rtype: bool
    """
    try:
        float(val)
    except ValueError:
        return False
    return True

def _token_to_mean(token, inputdict={}):
    """
    Convert an individual token to a mean function

    Function to convert a single non-operator token to a mean function.
    Makes substitutions found in the provided dictionary (if present)
    to map variable names to indices. Converts numeric tokens to ``ConstantMean``
    objects, tokens that can be converted into inputs to ``LinearMean`` objects,
    and any other strings not found inputdict are assumed to represent
    coefficients and will be converted to ``Coefficient`` objects.

    :param token: string token to be converted
    :type: token: str
    :param inputdict: dictionary used to map variables to input indices. Maps
                      strings to integer indices (must be non-negative). Optional,
                      default is ``{}``.
    :type inputdict: dict
    :returns: New subclass of ``MeanBase`` implementing the given token
    :rtype: subclass of MeanBase (exact type will depend on the token)
    """

    assert isinstance(token, str), "formula input to mean function is not a string"

    if _is_float(token):
        return MeanFunction.ConstantMean(float(token))

    token = _convert_token(token, inputdict)

    if not token[0] == "x":
        return MeanFunction.Coefficient()

    if not (token[:2] == "x[" and token[-1] == "]"):
        raise ValueError("bad formula input in mean function")

    try:
        index = int(token[2:-1])
    except ValueError:
        raise ValueError("index in parsed formula is not an integer")

    assert index >= 0, "index in formula parsing must be non-negative"

    return MeanFunction.LinearMean(index)

def _tokenize_string(formula):
    """
    Converts a string formula into a series of tokens for evaluation

    Takes a string and divides it into a list of individual string tokens.
    Special tokens include ``"(", ")", "+", "^", "[", "]", "=", "~"``,
    where these have the expected Python behavior, with ``"="`` and ``"~"``
    separating the LHS and RHS of the formula. All other special characters
    are considered to be non-special in this language and will only be split
    up based on whitespace. The LHS and all white space are removed. Note
    also that ``"call"`` is a protected keyword here, as it is used to
    designate a function call operation.

    :param formula: string formula to be parsed
    :type formula: str
    :returns: list of parsed string tokens
    :rtype: list
    """

    assert isinstance(formula, str)

    token_list = []
    accumulated = ""

    for char in formula:
        if char in ["(", ")", "+", "^", " ", "[", "]", "=", "~"]:
            if not accumulated == "":
                token_list.append(accumulated)
            token_list.append(char)
            accumulated = ""
        elif char == "*":
            if accumulated == "*":
                token_list.append("^")
                accumulated = ""
            elif not accumulated == "":
                token_list.append(accumulated)
                accumulated = "*"
            else:
                accumulated = "*"
        else:
            if accumulated == "*":
                token_list.append(accumulated)
                accumulated = ""
            accumulated += char

    if not accumulated == "":
        token_list.append(accumulated)

    outlist = []

    for item in token_list:
        if not item in [" ", "[", "]"]:
            outlist.append(item)
        elif item == "[":
            outlist.append(outlist.pop()+item)
        elif item == "]":
            if len(outlist) < 2:
                raise SyntaxError("error in using square brackets in formula input")
            outlist.append(outlist.pop(-2)+outlist.pop()+item)

    if outlist[0] == "y":
        outlist.pop(0)
    if outlist[0] in ["=", "~"]:
        outlist.pop(0)

    for item in outlist:
        if (not "[" in item and "]" in item) or (not "]" in item and "[" in item):
            raise SyntaxError("cannot nest operators in square brackets in formula input")
        if item == "call":
            raise SyntaxError("'call' cannot be used as a variable name in formula input")
        if item in ["=", "~"]:
            raise SyntaxError("LHS in formula is not correctly specified")

    return outlist

def _parse_tokens(token_list):
    """
    Parses a list of tokens, converting into an RPN sequence of operations

    Takes a list of string tokens and returns a list of string tokens only
    containing binary operators (``"+", "*", "^", "call"``) and objects on
    which to operate. Order of operations is denoted by Reverse Polish
    Notation, where operations take place on a stack. Uses an adapted
    shunting-yard algorithm.

    :param token_list: list of tokens to be parsed into a sequence of operations
    :type token_list: list
    :returns: list of operators and objects on which to operate
    :rtype: list
    """

    assert isinstance(token_list, list), "input must be a list of strings"

    prev_op = True

    operator_stack = []
    output_list = []

    precedence = {"+": 2, "*": 3, "^": 4}
    l_assoc = {"+": True, "*": True, "^": False}

    for token in token_list:
        assert isinstance(token, str), "input must be a list of strings"
        if token in ["=", "~"]:
            raise SyntaxError("LHS in formula is not correctly specified")
        if not token in ["(", ")", "+", "*", "^"]:
            output_list.append(token)
            prev_op = False
        if token == "(" and not prev_op:
            # function call, put call on the stack
            operator_stack.append("call")
            prev_op = True
        if token in ["+", "*", "^"]:
            while (len(operator_stack) >= 1 and
                   (not operator_stack[-1] == "(") and
                   (operator_stack[-1] == "call" or
                    precedence[operator_stack[-1]] > precedence[token] or
                    (precedence[operator_stack[-1]] == precedence[token] and l_assoc[token]))):
                    output_list.append(operator_stack.pop())
            operator_stack.append(token)
            prev_op = True
        if token == "(":
            operator_stack.append(token)
            prev_op = True
        if token == ")":
            while not operator_stack[-1] == "(":
                output_list.append(operator_stack.pop())
                if len(operator_stack) == 0:
                    raise SyntaxError("string expression has mismatched parentheses")
            if operator_stack[-1] == "(":
                operator_stack.pop()
            prev_op = False

    while len(operator_stack) > 0:
        operator = operator_stack.pop()
        if operator == ")" or operator == "(":
            raise SyntaxError("string expression has mismatched parentheses")
        output_list.append(operator)

    return output_list

def _eval_parsed_tokens(token_list, inputdict={}):
    """
    Evaluate parsed tokens into a mean function

    Evaluates the parsed token list, returning the mean function represented by these
    operations. Input must be a list of strings that come from the ``parse_tokens``
    function, though it will accept any list of strings for processing.

    The function evaluates the sequence of operators and objects on which the operations
    are performed by placing them on a stack and evaluating assuming Reverse Polish
    Notation. The objects are converted to mean function objects when they are placed on
    the stack, and if the stack ever does not contain a valid sequence of operations,
    a ``SyntaxError``is raised. Additionally, if at the end of all operations there is not
    a single mean function object on the stack, a ``SyntaxError`` is raised.

    :param token_list: list of tokens to be parsed into a sequence of operations
    :type token_list: list
    :param inputdict: dictionary used to map variables to input indices. Maps
                      strings to integer indices (must be non-negative). Optional,
                      default is ``{}``.
    :type inputdict: dict
    :returns: New subclass of ``MeanBase`` implementing the given set of operations
    :rtype: subclass of MeanBase (exact type will depend on the sequence)
    """

    assert isinstance(token_list, list), "input must be a list of strings"

    op_list = ["+", "*", "^", "call"]

    stack = []

    for token in token_list:
        assert isinstance(token, str), "tokens must be strings"
        if token in ["=", "~"]:
            raise SyntaxError("LHS in formula is not correctly specified")
        if token not in op_list:
            if token == "I":
                mf = "I"
            else:
                mf = _token_to_mean(token, inputdict)
            stack.append(mf)
        else:
            if len(stack) < 2:
                raise SyntaxError("string expression is not a valid mathematical expression")

            op_2 = stack.pop()
            if op_2 == "I":
                raise SyntaxError("identity operator can only be called as a function")
            assert issubclass(type(op_2), MeanFunction.MeanBase)
            op_1 = stack.pop()
            if token == "call":
                assert op_1 == "I" or issubclass(type(op_1), MeanFunction.MeanBase)
            else:
                if op_1 == "I":
                    raise SyntaxError("identity operator can only be called as a function")
                assert issubclass(type(op_1), MeanFunction.MeanBase)

            if token == "+":
                stack.append(op_1.__add__(op_2))
            elif token == "*":
                stack.append(op_1.__mul__(op_2))
            elif token == "^":
                stack.append(op_1.__pow__(op_2))
            elif token == "call":
                if op_1 == "I":
                    stack.append(op_2)
                else:
                    stack.append(op_1.__call__(op_2))
            else:
                raise SyntaxError("string expression is not a valid mathematical expression")

    if not len(stack) == 1:
        raise SyntaxError("string expression is not a valid mathematical expression")

    return stack[0]


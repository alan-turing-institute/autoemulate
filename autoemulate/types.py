from collections.abc import Iterable
from typing import Any
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import Self
from typing import TYPE_CHECKING
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

Matrixlike = Union[np.ndarray, np.matrix, Iterable[Iterable[float]]]
ArrayLike

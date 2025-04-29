from autoemulate.experimental.learners.base import Emulator

from . import stream

# from .base import Emulator, Simulator
from .base import Simulator

# # TODO: automate this with Learner.hierarchy()
# # lowercase classes below should correspond to abstract classesß
# class active:
#     class stream:
#         Random = _stream.Random
#         class threshold:

#             # Operate on the input space X
#             class input:
#                 Distance = _stream.Distance

#             # Operate on the output space Y, Sigma
#             class output:
#                 A_Optimal=_stream.A_Optimal
#                 D_Optimal=_stream.D_Optimal
#                 E_Optimal=_stream.E_Optimal

#             # TODO: add hybrid class (considers X, Y, Sigma)

#             # Adaptive versions of the above
#             class adaptive:
#                 class input:
#                     Distance=_stream.Adaptive_Distance
#                 class output:
#                     A_Optimal=_stream.Adaptive_A_Optimal
#                     D_Optimal=_stream.Adaptive_D_Optimal
#                     E_Optimal=_stream.Adaptive_E_Optimal

__all__ = ["Emulator", "Simulator", "stream"]

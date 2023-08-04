from abc import ABC, abstractmethod
import mogp_emulator

class ExperimentalDesign(ABC):
      def __init__(self, bounds_list):
            """Initializes a Sampler object.

            :param bounds_list: List tuples with two numeric values. 
                                Each tuple corresponds to the lower and 
                                upper bounds of a parameter.
            :type bounds_list: list
            """
            pass
      
      @abstractmethod
      def sample(self, n: int):
            """Samples n points from the sample space.

            :param n: The number of points to sample.
            :type n: int
            """
            pass
      
      @abstractmethod
      def get_n_parameters(self):
            """Returns the number of parameters in the sample space.

            :return: The number of parameters in the sample space.
            :rtype: int
            """
            pass

class LatinHypercube(ExperimentalDesign):
      def __init__(self, bounds_list):
            """Initializes a LatinHypercube object.

            :param bounds_list: List tuples with two numeric values. 
                                Each tuple corresponds to the lower and 
                                upper bounds of a parameter.
            :type bounds_list: list
            """
            self.sampler = mogp_emulator.LatinHypercubeDesign(bounds_list)
      
      def sample(self, n: int):
            """Samples n points from the sample space.

            :param n: The number of points to sample.
            :type n: int
            :return: A numpy array of shape (n, dim) containing the sampled points.
            :rtype: numpy.ndarray
            """
            return self.sampler.sample(n)

      def get_n_parameters(self):
            """Returns the number of parameters in the sample space.

            :return: The number of parameters in the sample space.
            :rtype: int
            """
            return self.sampler.get_n_parameters()
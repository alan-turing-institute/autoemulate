from abc import ABC, abstractmethod
import mogp_emulator

class Emulator(ABC):
      """An abstract base class for emulators.

      :param ABC: An abstract base class.
      """
      def __init__(self, sim_in, sim_out):
            """Initializes an Emulator object.

            :param sim_in: A numpy array of shape (n, dim) containing the input
                            parameters of the simulator.
            :type sim_in: numpy.ndarray
            :param sim_out: A numpy array of shape (n, dim) containing the output
                            of the simulator.
            :type sim_out: numpy.ndarray
            """
            pass
      
      def fit(self):
            """Fits the emulator to the data."""
            pass
      
      def predict(self):
            """Predicts the output of the simulator for a given input."""
            pass
      
class GaussianProcess(Emulator):
      """Gaussian process Emulator.
      
      """
      def __init__(self, sim_in, sim_out):
            """Initializes a GaussianProcess object.
            """
            self.gp = mogp_emulator.GaussianProcess(sim_in, sim_out, nugget='fit')
            
      def fit(self):
            """Fits the emulator to the data."""
            return self.gp.fit_GP_MAP()
      
      def predict(self):
            """Predicts the output of the simulator for a given input.
            """
            return self.gp.predict()
            
            
from abc import ABC, abstractmethod
from sklearn.svm import SVR   
import mogp_emulator

class Emulator(ABC):
      """An abstract base class for emulators.

      """
      @abstractmethod
      def __init__(self, *args, **kwargs):
            """Initializes an Emulator object.
            """
            pass
      
      @abstractmethod
      def fit(self, X, y):
            """Fits the emulator to the data.
            
            :param X: Input data (simulation input).
            :param y: Target data (simulation output). 
            """
            pass
      
      @abstractmethod
      def predict(self, X):
            """Predicts the output of the simulator for a given input."""
            pass
      
class GaussianProcess(Emulator):
      """Gaussian process Emulator.
      
      Implements GaussianProcsses from the mogp_emulator package. 
      """
      def __init__(self, *args, **kwargs):
            """Initializes a GaussianProcess object."""
            self.args = args
            self.kwargs = kwargs
            self.model = None
           
      def fit(self, X, y):
            """Fits the emulator to the data.
            
            :param X: Input data (simulation input).
            :param y: Target data (simulation output). 
            """
            self.model = mogp_emulator.GaussianProcess(X, y, nugget='fit', 
                                                       *self.args, **self.kwargs)
            self.model = mogp_emulator.fit_GP_MAP(self.model)
      
      def predict(self, X):
            """Predicts the output of the simulator for a given input.
            
            :param X: Input data (simulation input).
            """
            if self.model is not None:
                  return self.model.predict(X)
            else:
                  raise ValueError("Emulator not fitted yet.")

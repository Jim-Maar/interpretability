from abc import ABC, abstractmethod

class NeuronPredictor(ABC):
    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass
    
    @abstractmethod
    def get_clean_format(self):
        pass

    @abstractmethod
    def get_sparcity(self):
        pass

    @abstractmethod
    def load(self, layer, neuron):
        """
        Load the model from a file
        return True if the model was loaded successfully
        """
        pass

    @abstractmethod
    def save(self):
        pass
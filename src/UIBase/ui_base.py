from abc import ABC, abstractmethod
from typing import Any

class UIBase(ABC):
    """
    Abstract base class for UI components.
    
    This class serves as a blueprint for all UI classes in the application. 
    Derived classes should implement the 'initialize' and 'interface' methods.
    """

    @abstractmethod
    def initialize(self) -> Any:
        """
        Initialize the UI component.

        Returns:
            Any: The initialized component. The type can vary based on the subclass implementation.
        """
        pass

    @abstractmethod
    def interface(self) -> Any:
        """
        Generate the interface for the UI component.

        Returns:
            Any: The user interface component that can be used within a larger Gradio-based app.
        """
        pass
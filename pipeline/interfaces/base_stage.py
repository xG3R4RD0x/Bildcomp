from abc import ABC, abstractmethod
from typing import Any, Dict


class CompressionStage(ABC):
    """
    --Compression Stage Interface--
    This Interface defines the structure for each compression stage in pipeline.

    Each stage has to implement:
    - name: A method to return the identifier name of the stage.
    - process: A method to process the data and return the transformed results.


    """

    @abstractmethod
    def name(self) -> str:
        """
        name returns the name of the stage.
        This name is going to be useful for logging and debugging
        """
        pass

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        this is the main method of the stage.
        It processes the data and returns the transformed results.
        It accepts a dictionary as input and returns a dictionary as output.
        """
        pass

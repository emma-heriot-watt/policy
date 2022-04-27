from abc import ABC, abstractmethod
from typing import Any, Optional

from PIL.Image import Image
from pydantic import BaseModel


class SimulatorAction(BaseModel):
    """Dictionary containing the previous action taken by the agent.

    Args:
        action: Action taken by the agent in the environment.
        obj_relative_coord: Relative (x, y) coordinate indicating the object in the image.

    Note:   The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the
            agent's egocentric image, selects an object in a 10x10 pixel patch around the pixel
            indicated by the coordinate if the desired action can be performed on it, and executes
            the action in AI2-THOR.
    """

    action: Optional[str]
    obj_relative_coord: Optional[tuple[float, float]]


class BaseModelWrapper(ABC):
    """Base wrapper to use so that the model can communicate with the inference engine.

    This has been implemented in line with `teach.inference.teach_model.TeachModel` from the
    `alexa/teach` repo: https://github.com/alexa/teach/blob/main/src/teach/inference/teach_model.py

    Documentation included any additional questions/discoveries found during implementation of the
    inference engine.
    """

    @abstractmethod
    def __init__(self, process_index: int, num_processes: int, model_args: list[str]) -> None:
        """A model will be initialized for each evaluation process.

        Args:
            process_index: Index of the process that LAUNCHED the model
            num_processes: Total number of processes that are launched in parallel
            model_args: Extra CLI arguments to `teach_eval` which get passed to the model.
                How relevant is this for our model?
        """

    @abstractmethod
    def get_next_action(
        self,
        img: Image,
        edh_instance: Any,
        prev_action: Optional[SimulatorAction],
        img_name: Optional[str] = None,
        edh_name: Optional[str] = None,
    ) -> tuple[str, Optional[tuple[float, float]]]:
        """Get the next predicted action from the model.

        Called at each timestep.

        Args:
            img: Agent's egocentric image.
            edh_instance: EDH Instance from the file
            prev_action: Previous action taken by the agent, if any
            img_name: File name of the image
            edh_name: File name for the EDH instance

        Returns:
            - action name from `all_agent_actions`
            - obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1)
                indicating an object in the image
        """
        raise NotImplementedError

    @abstractmethod
    def start_new_edh_instance(
        self,
        edh_instance: Any,
        edh_history_images: list[Image],
        edh_name: Optional[str] = None,
    ) -> bool:
        """Start a new EDH instance, resetting the state and anything else.

        Called at the start of each EDH instance AFTER the environment has ben set to the initial
        state, but before actions are requested from the model.

        Args:
            edh_instance: EDH Instance from the file
            edh_history_images: Images loaded from the files specified in
                `edh_instance['driver_image_history']`
            edh_name: File name for the EDH instance

        Returns:
            True if successfully created an EDH instance
        """
        raise NotImplementedError

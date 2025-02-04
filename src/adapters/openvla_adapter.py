from typing import Any, Dict, Optional
from ..core.adapter import Adapter
import numpy as np

class OpenVLAAdapter(Adapter):
    """Adapter for integrating OpenVLA models into EmbodyHub.
    
    This adapter provides compatibility with OpenVLA's vision-language-action models,
    handling the conversion between OpenVLA's specific formats and EmbodyHub's
    standardized formats.
    """
    
    def _initialize(self) -> None:
        """Initialize OpenVLA-specific components.
        
        This method sets up any necessary configurations and validates
        required dependencies.
        """
        self.input_modalities = self.config.get('input_modalities', ['vision', 'language'])
        self.output_modalities = self.config.get('output_modalities', ['action'])
    
    def convert_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenVLA observation format to EmbodyHub format.
        
        Args:
            observation: A dictionary containing observations in OpenVLA format.
                Expected keys include 'visual_input', 'text_input', etc.
        
        Returns:
            A dictionary containing the standardized observation format.
        """
        converted_obs = {}
        
        if 'vision' in self.input_modalities and 'visual_input' in observation:
            # Convert visual input (assuming numpy array format)
            converted_obs['visual'] = np.array(observation['visual_input'])
        
        if 'language' in self.input_modalities and 'text_input' in observation:
            # Convert text input
            converted_obs['text'] = str(observation['text_input'])
        
        return converted_obs
    
    def convert_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert EmbodyHub action format to OpenVLA format.
        
        Args:
            action: A dictionary containing actions in EmbodyHub format.
                Expected keys depend on the configured output modalities.
        
        Returns:
            A dictionary containing the action in OpenVLA's format.
        """
        converted_action = {}
        
        if 'action' in self.output_modalities:
            if 'command' in action:
                converted_action['action_command'] = action['command']
            if 'parameters' in action:
                converted_action['action_params'] = action['parameters']
        
        return converted_action
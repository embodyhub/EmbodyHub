from typing import Any, Dict, Optional
from ..core.adapter import Adapter
import numpy as np

class OCTOAdapter(Adapter):
    """Adapter for integrating OCTO models into EmbodyHub.
    
    This adapter provides compatibility with OCTO's multimodal models,
    handling the conversion between OCTO's specific formats and EmbodyHub's
    standardized formats.
    """
    
    def _initialize(self) -> None:
        """Initialize OCTO-specific components.
        
        This method sets up any necessary configurations and validates
        required dependencies.
        """
        self.supported_modalities = self.config.get('modalities', ['image', 'text', 'audio'])
        self.model_type = self.config.get('model_type', 'multimodal')
        self.device = self.config.get('device', 'cuda')
    
    def convert_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OCTO observation format to EmbodyHub format.
        
        Args:
            observation: A dictionary containing observations in OCTO format.
                Expected keys include 'image_data', 'text_data', 'audio_data', etc.
        
        Returns:
            A dictionary containing the standardized observation format.
        """
        converted_obs = {}
        
        # Handle image modality
        if 'image' in self.supported_modalities and 'image_data' in observation:
            # Convert image data (assuming numpy array format)
            converted_obs['image'] = np.array(observation['image_data'])
        
        # Handle text modality
        if 'text' in self.supported_modalities and 'text_data' in observation:
            converted_obs['text'] = str(observation['text_data'])
        
        # Handle audio modality
        if 'audio' in self.supported_modalities and 'audio_data' in observation:
            converted_obs['audio'] = np.array(observation['audio_data'])
        
        return converted_obs
    
    def convert_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert EmbodyHub action format to OCTO format.
        
        Args:
            action: A dictionary containing actions in EmbodyHub format.
                Expected keys depend on the configured modalities.
        
        Returns:
            A dictionary containing the action in OCTO's format.
        """
        converted_action = {}
        
        # Convert based on model type
        if self.model_type == 'multimodal':
            converted_action['model_output'] = {
                'modality': action.get('modality', 'text'),
                'content': action.get('content', {}),
                'metadata': action.get('metadata', {})
            }
        
        return converted_action
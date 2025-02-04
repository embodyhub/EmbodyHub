from typing import Dict, Any
from src.adapters.openvla_adapter import OpenVLAAdapter
import numpy as np

def main():
    # Initialize the OpenVLA adapter with custom configurations
    config = {
        'input_modalities': ['vision', 'language'],
        'output_modalities': ['action']
    }
    adapter = OpenVLAAdapter(config)
    
    # Simulate a visual-language observation
    mock_observation = {
        'visual_input': np.random.rand(224, 224, 3),  # Mock RGB image
        'text_input': "Pick up the red cup from the table"
    }
    
    # Convert observation to EmbodyHub format
    embodyhub_obs = adapter.convert_observation(mock_observation)
    print("\nConverted Observation:")
    print(f"Visual shape: {embodyhub_obs['visual'].shape}")
    print(f"Text input: {embodyhub_obs['text']}")
    
    # Simulate an action in EmbodyHub format
    embodyhub_action = {
        'command': 'pick_and_place',
        'parameters': {
            'object': 'red_cup',
            'source': 'table',
            'destination': 'hand'
        }
    }
    
    # Convert action to OpenVLA format
    openvla_action = adapter.convert_action(embodyhub_action)
    print("\nConverted Action:")
    print(f"Action Command: {openvla_action['action_command']}")
    print(f"Action Parameters: {openvla_action['action_params']}")

if __name__ == '__main__':
    main()
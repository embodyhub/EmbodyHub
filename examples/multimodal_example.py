"""Multimodal Data Management Example

This example demonstrates how to use the MultimodalDataManager to handle
different types of data streams in an embodied AI application.
"""

import numpy as np
import time
from threading import Thread

from embodyhub.adapters.multimodal_data_manager import MultimodalDataManager

def generate_image_data():
    """Generate simulated image data."""
    return np.random.rand(64, 64, 3)  # RGB image

def generate_audio_data():
    """Generate simulated audio data."""
    return np.random.rand(16000)  # 1 second of audio at 16kHz

def generate_sensor_data():
    """Generate simulated sensor data."""
    return np.random.rand(10)  # 10 sensor readings

def data_callback(data: np.ndarray):
    """Callback function for processing received data."""
    print(f"Received data shape: {data.shape}, dtype: {data.dtype}")

def main():
    # Initialize data manager
    data_manager = MultimodalDataManager()
    
    # Create data streams
    data_manager.create_stream('camera_feed')
    data_manager.create_stream('microphone_input')
    data_manager.create_stream('sensor_readings')
    
    # Subscribe to streams
    data_manager.subscribe('camera_feed', data_callback)
    data_manager.subscribe('microphone_input', data_callback)
    data_manager.subscribe('sensor_readings', data_callback)
    
    # Start data processing
    data_manager.start()
    
    # Simulate data generation
    def data_generator():
        for _ in range(10):  # Generate 10 samples
            # Publish image data
            image_data = generate_image_data()
            data_manager.publish('camera_feed', image_data)
            
            # Publish audio data
            audio_data = generate_audio_data()
            data_manager.publish('microphone_input', audio_data)
            
            # Publish sensor data
            sensor_data = generate_sensor_data()
            data_manager.publish('sensor_readings', sensor_data)
            
            time.sleep(1)  # Wait 1 second between samples
    
    # Run data generation in a separate thread
    generator_thread = Thread(target=data_generator)
    generator_thread.start()
    
    # Wait for data generation to complete
    generator_thread.join()
    
    # Stop data manager
    data_manager.stop()

if __name__ == '__main__':
    main()
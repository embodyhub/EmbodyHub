# EmbodyHub Examples

This directory contains various examples of using the EmbodyHub framework, helping you quickly get started and understand the core functionalities.

## Example List

### 1. PyTorch Integration Example (pytorch_example.py)

This example demonstrates how to integrate reinforcement learning models with EmbodyHub using the PyTorch adapter:
- Creating a simple policy network
- Implementing custom environments
- Using PyTorchAdapter to register and manage models

### 2. Multimodal Data Management Example (multimodal_example.py)

Demonstrates how to handle different types of data streams using MultimodalDataManager:
- Creating and managing multiple data streams
- Processing image, audio, and sensor data
- Implementing data callback handling

### 3. OpenVLA Integration Example (openvla_example.py)

Shows how to handle vision-language tasks using the OpenVLA adapter:
- Configuring vision-language input/output modalities
- Converting observation and action data
- Implementing vision-language guided task execution

### 4. Performance Optimization Example (optimization_example.py)

Demonstrates the use of EmbodyHub's performance optimization tools:
- Model quantization and optimization
- Memory management optimization
- Parallel execution optimization

## Running Examples

1. Ensure all dependencies are installed:
```bash
pip install -r ../requirements.txt
```

2. Run specific examples:
```bash
python pytorch_example.py
# or
python multimodal_example.py
# or
python openvla_example.py
# or
python optimization_example.py
```
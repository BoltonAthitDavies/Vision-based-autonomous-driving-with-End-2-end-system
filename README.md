# Vision-based Autonomous Driving with End-2-End System

Vision-based autonomous driving project for FRA361 (Open Topic). This repository provides a complete end-to-end learning pipeline that maps camera images directly to driving commands, focusing on feasibility, computational aspects, and limitations of vision-only autonomous driving.

## 🚗 Overview

This project implements an end-to-end deep learning approach for autonomous driving, inspired by NVIDIA's PilotNet architecture. The system learns to predict steering angles (and other driving commands) directly from raw camera images, without the need for hand-crafted features or explicit perception modules.

### Key Features

- **End-to-End Learning**: Direct mapping from images to driving commands
- **PilotNet Architecture**: Implementation of NVIDIA's proven CNN architecture
- **Flexible Data Pipeline**: Support for custom datasets with CSV annotations
- **Comprehensive Training Tools**: Training, evaluation, and visualization utilities
- **Configurable**: YAML-based configuration for easy experimentation
- **Modular Design**: Clean, extensible codebase for research and development

## 📁 Project Structure

```
Vision-based-autonomous-driving-with-End-2-end-system/
├── config/
│   └── config.yaml              # Training configuration
├── data/
│   ├── raw/                     # Raw image data
│   └── processed/               # Processed data
├── models/                      # Saved model weights
├── notebooks/                   # Jupyter notebooks for exploration
├── src/
│   ├── data/
│   │   ├── dataset.py          # Dataset classes
│   │   └── transforms.py       # Data augmentation
│   ├── models/
│   │   ├── base_model.py       # Base model interface
│   │   └── pilotnet.py         # PilotNet architecture
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   └── evaluator.py        # Evaluation utilities
│   └── utils/
│       ├── visualization.py     # Plotting functions
│       ├── logger.py           # Logging utilities
│       └── checkpoint.py       # Model checkpointing
├── tests/                       # Unit tests
├── train.py                     # Main training script
├── inference.py                 # Inference/evaluation script
├── demo.py                      # Demo script
└── requirements.txt             # Python dependencies
```

## 🔧 Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/BoltonAthitDavies/Vision-based-autonomous-driving-with-End-2-end-system.git
cd Vision-based-autonomous-driving-with-End-2-end-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Format

The system expects data in the following format:

### CSV File Structure
```csv
image_path,steering_angle,throttle,brake,speed
data/raw/image_0001.jpg,0.15,0.5,0.0,25.3
data/raw/image_0002.jpg,-0.08,0.5,0.0,25.5
...
```

### Directory Structure
```
data/
├── train.csv          # Training data annotations
├── val.csv            # Validation data annotations
├── test.csv           # Test data annotations
└── raw/              # Directory containing images
    ├── image_0001.jpg
    ├── image_0002.jpg
    └── ...
```

## 🚀 Usage

### 1. Run Demo (No Data Required)

See the system in action without any dataset:

```bash
python demo.py
```

This will demonstrate:
- Model creation and architecture
- Data preprocessing
- Inference pipeline
- Training setup

### 2. Training

Update the configuration file `config/config.yaml` with your data paths, then:

```bash
python train.py --config config/config.yaml
```

**Training Options:**
- Modify `config/config.yaml` to adjust hyperparameters
- Change model architecture (PilotNet or SimplePilotNet)
- Configure data augmentation
- Set up learning rate scheduling

### 3. Evaluation

Evaluate a trained model on test data:

```bash
python inference.py --model_path models/best_model.pth --visualize --save_predictions
```

**Inference Options:**
- `--model_path`: Path to trained model weights
- `--visualize`: Generate prediction plots
- `--save_predictions`: Save predictions to file

### 4. Custom Usage

```python
import torch
from src.models.pilotnet import PilotNet
from src.data.transforms import get_val_transforms
from PIL import Image

# Load model
model = PilotNet(input_channels=3, output_size=1)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Prepare image
transform = get_val_transforms(input_size=(66, 200))
image = Image.open('test_image.jpg')
image_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    steering_angle = model(image_tensor)
    print(f"Predicted steering angle: {steering_angle.item():.4f}")
```

## 🏗️ Model Architecture

### PilotNet

Based on NVIDIA's end-to-end learning architecture:

- **Input**: 66x200 RGB image
- **Conv Layers**: 5 convolutional layers with batch normalization
  - Conv1: 24 filters (5x5, stride 2)
  - Conv2: 36 filters (5x5, stride 2)
  - Conv3: 48 filters (5x5, stride 2)
  - Conv4: 64 filters (3x3, stride 1)
  - Conv5: 64 filters (3x3, stride 1)
- **FC Layers**: 4 fully connected layers
  - FC1: 100 neurons
  - FC2: 50 neurons
  - FC3: 10 neurons
  - FC4: 1 output (steering angle)
- **Regularization**: Dropout and batch normalization

### SimplePilotNet

A lightweight version for faster training and testing.

## 📈 Training Configuration

Key configuration options in `config/config.yaml`:

```yaml
model:
  type: "PilotNet"
  output_size: 1
  dropout_rate: 0.5

training:
  num_epochs: 50
  learning_rate: 0.001
  batch_size: 64
  optimizer: "adam"
  loss_function: "mse"

data:
  image_size: [66, 200]
  target_columns: ["steering_angle"]
```

## 🔬 Features

### Data Augmentation
- Random brightness adjustment
- Color jitter
- Horizontal flipping
- Custom shadow simulation

### Training Features
- Automatic checkpointing
- Learning rate scheduling
- Early stopping support
- Training visualization
- TensorBoard integration (optional)

### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Error distribution analysis

## 🎯 Use Cases

This template can be used for:
- **Simulation Training**: Train models in driving simulators (CARLA, AirSim, etc.)
- **Real-world Transfer**: Fine-tune on real driving data
- **Research**: Experiment with different architectures and training strategies
- **Education**: Learn about end-to-end autonomous driving systems

## 📝 Notes and Limitations

- This is a **template implementation** focused on the software pipeline
- Requires substantial training data for real-world applications
- Vision-only approach has inherent limitations (weather, lighting, occlusions)
- Should be combined with other sensors (LiDAR, radar) for production systems
- Extensive testing and validation required for safety-critical applications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is provided as-is for educational and research purposes.

## 📚 References

- [NVIDIA PilotNet Paper](https://arxiv.org/abs/1604.07316): End-to-End Learning for Self-Driving Cars
- PyTorch Documentation
- Modern autonomous driving research

## 🔗 Related Resources

- [CARLA Simulator](https://carla.org/)
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)
- [AirSim](https://github.com/microsoft/AirSim)

---

**Note**: This is a template/reference implementation for educational purposes. For production autonomous driving systems, extensive testing, validation, and integration with additional sensors and safety systems is required.

# Data Directory

This directory contains the datasets for training, validation, and testing.

## Expected Structure

```
data/
├── raw/                    # Raw image files
│   ├── image_0001.jpg
│   ├── image_0002.jpg
│   └── ...
├── processed/             # Processed/augmented data (optional)
├── train.csv              # Training data annotations
├── val.csv                # Validation data annotations
└── test.csv               # Test data annotations
```

## CSV Format

Each CSV file should contain the following columns:

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| image_path | Relative path to image | string | `raw/image_0001.jpg` |
| steering_angle | Steering angle in radians | float | `0.15` |
| throttle | Throttle value (0-1) | float | `0.5` |
| brake | Brake value (0-1) | float | `0.0` |
| speed | Speed in m/s or km/h | float | `25.3` |

### Example CSV Content

```csv
image_path,steering_angle,throttle,brake,speed
raw/image_0001.jpg,0.15,0.5,0.0,25.3
raw/image_0002.jpg,-0.08,0.5,0.0,25.5
raw/image_0003.jpg,0.00,0.6,0.0,26.1
```

## Data Collection

You can collect data from:

1. **Driving Simulators**:
   - CARLA Simulator
   - Udacity Self-Driving Car Simulator
   - AirSim
   - TORCS

2. **Real-world Driving**:
   - Front-facing camera recordings
   - CAN bus data for steering/throttle/brake
   - GPS and IMU data (optional)

3. **Public Datasets**:
   - Udacity Self-Driving Car Dataset
   - comma.ai Dataset
   - Berkeley DeepDrive (BDD100K)

## Data Preprocessing

The system automatically handles:
- Image resizing to 66x200 (PilotNet input size)
- Normalization using ImageNet statistics
- Data augmentation (for training only)

## Data Split Recommendations

- **Training**: 70-80% of data
- **Validation**: 10-15% of data
- **Testing**: 10-15% of data

Ensure data is well-distributed across:
- Different lighting conditions
- Various road types
- Different weather conditions
- Range of steering angles

## Notes

- Images should be in RGB format (JPG or PNG)
- Steering angles typically range from -1 (full left) to +1 (full right)
- Ensure consistent frame rate and synchronization between images and labels
- Balance your dataset to avoid bias towards straight driving

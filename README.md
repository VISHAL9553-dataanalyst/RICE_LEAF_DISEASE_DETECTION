# Rice Leaf Disease Detection

## Overview
This project focuses on detecting and classifying rice leaf diseases using image-based machine learning techniques. Leveraging Convolutional Neural Networks (CNN) and Transfer Learning (VGG16 and InceptionV3), the model aims to identify diseases such as Leaf Smut, Brown Spot, and Bacterial Leaf Blight from rice leaf images. The solution is designed to assist farmers and researchers in diagnosing diseases effectively and efficiently.

---

## Features
The project handles the classification of three types of rice leaf diseases:

1. **Leaf Smut**: Caused by the fungus *Entyloma oryzae*, characterized by black spots on leaves.
2. **Brown Spot**: A fungal disease affecting multiple parts of the rice plant, leading to brown spots on leaves and grains.
3. **Bacterial Leaf Blight**: A bacterial infection causing water-soaked lesions and streaks on leaves and petioles.

---

## Dataset
- **Source**: Rice leaf disease images provided in three categories (Leaf Smut, Brown Spot, Bacterial Leaf Blight).
- **Size**: 120 images (40 per class) in JPEG format.
- **Preprocessing**: Includes resizing, rescaling, data augmentation, and splitting into training, validation, and testing sets.

---

## Methodology
### Data Augmentation:
Techniques such as rotation, zooming, and flipping were applied to increase the diversity of training samples.

### Model Architecture:
1. **CNN (Custom)**: Initial model achieving ~55% validation accuracy.
2. **Transfer Learning**:
   - **InceptionV3**: Achieved ~69% validation accuracy.
   - **VGG16**: Achieved ~65% validation accuracy.

### Loss Function and Optimizer:
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam with a learning rate of 0.0002

---

## Project Structure
```plaintext
├── Rice_Leaf_Disease_Detection.py   # Main script for data preparation, training, and evaluation
├── Dataset
│   ├── Train                      # Training images organized by class
│   ├── Test                       # Testing images organized by class
├── README.md                       # Project documentation
├── Results
│   └── Evaluation_Results.md       # Model evaluation and metrics
```

---

## Setup and Installation

### Prerequisites
Ensure you have Python 3.8 or higher installed along with the required libraries.

### Libraries
The following libraries are used in this project:

- tensorflow
- keras
- numpy
- pandas
- matplotlib
- opencv-python
- pillow

### Installation
Clone the repository and install the required libraries:

```bash
$ git clone https://github.com/your-repository-url.git
$ cd rice-leaf-disease-detection
$ pip install -r requirements.txt
```

---

## Usage
1. Organize the dataset in the `Train` and `Test` folders under `Dataset`.
2. Run the script `Rice_Leaf_Disease_Detection.py` for data preprocessing, training, and evaluation:

```bash
$ python Rice_Leaf_Disease_Detection.py
```

3. Review training/validation accuracy and predictions for test images.

---

## Model Evaluation

The project evaluates the following models:
1. **Custom CNN**:
   - Validation Accuracy: ~55%
   - Training Accuracy: ~65%

2. **Transfer Learning Models**:
   - **InceptionV3**:
     - Validation Accuracy: ~69%
     - Training Accuracy: ~97%
   - **VGG16**:
     - Validation Accuracy: ~65%
     - Training Accuracy: ~98%

### Metrics Used:
- Training and Validation Accuracy
- Loss Function

---

## Results
- Validation accuracy improved significantly with InceptionV3 and VGG16 compared to the custom CNN.
- Data augmentation and hyperparameter tuning played a key role in optimizing model performance.

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request for any changes or improvements.

---

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## Acknowledgments
- Special thanks to the creators of the rice leaf disease dataset.
- Tools and libraries used include TensorFlow, Keras, and OpenCV for model training and image preprocessing.

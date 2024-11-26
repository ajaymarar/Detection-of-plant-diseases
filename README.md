This project implements a Vision Transformer model optimized with the ABC optimizer to detect diseases in plant leaves. The model is trained and tested on the PlantVillage dataset, achieving high performance metrics.

Requirements
To install the necessary packages, use the following command:

pip install -r requirements.txt
Project Structure
new - Copy.ipynb: The main Jupyter notebook containing the code for data processing, model training, and evaluation.
requirements.txt: List of dependencies required for the project.
Training
The dataset is split into three parts:

Training Set: 80% of the dataset is used for training the model.
Validation Set: 10% of the dataset is used for validating the model during training to tune hyperparameters and prevent overfitting.
Testing Set: 10% of the dataset is used for testing the model after training to evaluate its performance.
The training process includes the following steps:

Data Loading and Preprocessing: Load the images from the PlantVillage dataset, apply data augmentation techniques, and normalize the images.
Model Building: Construct the Vision Transformer model with the specified layers and compile it with the chosen optimizer and loss function.
Model Training: Train the model using the training set, validate it on the validation set, and monitor the performance using specified metrics.
Model Evaluation: Evaluate the final model on the testing set and compute performance metrics such as accuracy, precision, recall, and F1 score.
Vision Transformer Layers
Data Augmentation Layer
The Data Augmentation Layer applies a series of transformations such as random flipping (horizontally flipping the image), rotation (randomly rotating the image by a certain angle), and zooming (randomly zooming into the image) to increase the diversity of the training data. These augmentations enhance the model's ability to generalize by making it more robust to variations in the input data.

Patches Layer
In the Patches Layer, augmented images are divided into smaller patches, typically of size 16x16 or 32x32 pixels. Each patch represents a localized part of the image and is processed independently. This allows the model to focus on capturing local features like texture and small details, which are essential for accurate disease detection.

Patch Encoder Layer
The Patch Encoder Layer embeds each image patch into a higher-dimensional space, facilitating a more informative representation for subsequent layers. This embedding process captures spatial relationships and dependencies between different patches, aiding in the model's understanding of the overall image context.

Transformer Layer
At the core of the Vision Transformer model, the Transformer Layer employs a multi-head self-attention mechanism. This mechanism enables the model to capture relationships between different patches by attending to various parts of the image simultaneously. By doing so, the model gains insights into the global context of the image and how different parts relate to each other, thereby enhancing its ability to learn complex patterns.

MLP Layer
Following the Transformer Layer, the Multi-Layer Perceptron (MLP) processes the aggregated information. This fully connected layer integrates the encoded image patches to make final classification decisions. With multiple dense layers and activation functions, the MLP Layer effectively captures non-linear relationships between features, further refining the model's predictive capabilities.

ABC Optimizer Layer
The ABC Optimizer Layer fine-tunes the model using an optimization algorithm inspired by the foraging behavior of honey bees. It iteratively updates potential solutions based on the quality of simulated food sources, balancing exploration and exploitation to converge towards the global minimum efficiently. This approach optimizes model parameters, maximizing performance on the validation and testing datasets in this project context.

Evaluation
The model's performance is evaluated using various metrics and visualizations:

Classification Report: Provides a detailed report showing the precision, recall, F1 score, and accuracy.
Confusion Matrix: A matrix to visualize the performance of the classification model.
Prediction Visualizations: Visualizing the predictions made by the model.
Dataset
The dataset used for this project is the PlantVillage Dataset.

Results
The model achieves high performance on the testing dataset, demonstrating its effectiveness in detecting diseases in plant leaves.

The dataset used for this project is the PlantVillage Dataset. https://www.kaggle.com/datasets/emmarex/plantdisease

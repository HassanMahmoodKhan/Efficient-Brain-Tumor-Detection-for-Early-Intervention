# Deep-Convolutional-Neural-Networks-for-Medical-Image-Classification

This project aims to employ several deep learning architectures to conduct a thorough classification analysis of brain MRI scan images and classify them to one of the two classes i.e., tumor or no-tumor. The aim is to attempt to build upon the existing architectures by leveraging concepts such as transfer learning to improve them via custom additions. Finally, a comprehensive analysis is produced for each model to weigh the advantages and disadvantages, along with potential for future use.

## Convolutional Neural Networks
CNNs are a type of neural network architecture that can uncover key information in both time series and image data. For this reason, it is highly valuable for image-related tasks, such as image recognition, object classification and pattern recognition. Compared to the other networks, a CNN delivers better performance with image inputs, and with speech or audio signal inputs. A CNN consists of at least three layers: a convolutional layer, a pooling layer and a fully connected (FC) layer.

### Convolutional Layer

The convolutional layer is the core building block of a CNN, responsible for most computations. In this layer, a kernel or filter is convolved with the input image, sliding over the receptive fields of the image to detect features. This process creates a feature map or convolved feature, which is ultimately used to interpret the image and extract relevant patterns from it.

### Pooling Layer

The pooling layer is responsible for reducing the number of parameters in the input and improving the efficiency of the CNN. Like the convolutional layer, a kernel or filter is applied to the input image. However, instead of detecting features, the pooling layer aggregates information and downsamples the feature maps. This reduces the dimensionality of the data, resulting in computational savings and a more efficient network.

### Fully Connected Layer

The fully connected (FC) layer is where image classification happens in the CNN, based on the features extracted in the previous layers. In this layer, all the inputs from the previous layer are connected to every activation unit or node of the next layer. The FC layer is the final layer of the CNN, responsible for classifying the input image into a specific category.

## Transfer Learning/Fine Tuning

The idea is to adapt an existing pre-trained model to a different domain by executing a few training iterations on a new data set. The notion of fine-tuning is based on the intuition that low-level information/features can be reused in different applications, without training from scratch. Even when the final classification objective is different, it is also a relevant approach for initializing the learnable parameters close to good local minima, instead of initializing with random weights.

## Dataset

Brain MRI Images for Brain Tumor Detection. The dataset consists of 253 image samples of high-resolution brain MRI scans. The images are grayscale in nature and vary in size. The aim of the dataset is to provide evidence for conducting image analysis to predict whether each image belongs to the Tumor or Non-tumor category.

![image](https://github.com/HassanMahmoodKhan/Deep-Convolutional-Neural-Networks-for-Medical-Image-Classification/assets/97694796/9d18efa6-9c7e-4e80-87e4-e3eeb7d06508)

## Architecture

![image](https://github.com/HassanMahmoodKhan/Deep-Convolutional-Neural-Networks-for-Medical-Image-Classification/assets/97694796/214948ac-f73b-4a27-822b-aefd78f0fd6c)

## Data Preprocessing

I have employed the following steps to preprocess the dataset:
- Train & Test Split
- Data Augmentation
- Feature Scaling

## Model Building and Training

I have employed three deep convolutional neural networks that are considered benchmark models in the field of computer vision and in particular image classification:
- ResNet50V2
- VGG19
- VGG16

We look at how each of these models have performed under different scenarios using the evaluation metrics stated below and discuss their comparative performance. The results that have been obtained are used to develop an objective view of the overall efficacy of the architecture employed in this project.

## Model Evaluation

Evaluation metrics are used to measure the quality of the statistical or machine learning model. In this regard, for this binary classification problem, I have attempted to use a few metrics to gauge the performance of the various models:
- Accuracy
- Binary Cross Entrpy
- Confusion Matrix

## Inference Speed

We have performed inferencing for all models using the native Tensorflow implementation and compared it with the ONNX based model conversion, using ONNX Rutime. This was implemented to understand the speed-up for large & complex deep learning models and their viability for productionization under the pretext of high performance computing. Note: This was only performed using a 'CPU' execution provider.

![image](https://github.com/HassanMahmoodKhan/Deep-Convolutional-Neural-Networks-for-Medical-Image-Classification/assets/97694796/71b49147-9692-4690-a0af-db5f62665f2f)

## Results

The results obtained can be summarized by the table below:

![image](https://github.com/HassanMahmoodKhan/Deep-Convolutional-Neural-Networks-for-Medical-Image-Classification/assets/97694796/4a70ee7c-90e6-4bc3-95f2-d9788256e7c2)

- VGG19 with augmented training data has the highest test set classification score of 86%.
- VGG16 with augmented training data has the least BinaryCrossEntropy loss of 0.4984.


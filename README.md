# FASHION_MNIST_IMAGE_CLASSIFICATION

CSC120 IMAGE CLASSIFICATION


1. What is the Fashion MNIST dataset?
  - Fashion-MNIST is a dataset of fashion images that is intended as a direct replacement for the original MNIST dataset, which contains handwritten digits, for benchmarking machine learning algorithms. It has the same image size, data format, and split structure as the original MNIST dataset.

3. Why do we normalize image pixel values before training?
  - to enhance the efficiency and stability of the learning process. The raw pixels of an image are normally in the range of 0 to 255, which may lead to a large difference in the scale of the input values. Normalizing these values, normally scaling them to the range of 0 to 1 or -1 to 1, ensures that all features have an equal influence on the calculations of the model. This enhances the efficiency of the model by ensuring that the model trains faster, without some neurons being dominated by larger input values.
    
4. List the layers used in the neural network and their functions.
  - Input Layer – Receives the raw data (e.g., image pixels) and passes it to the network.
  - Dense (Fully Connected) Layer – Each neuron connects to all neurons in the previous layer; used to learn patterns and features.
  - Convolutional (Conv) Layer – Applies filters to detect local patterns, edges, or features in data like images.
  - Pooling Layer – Reduces the spatial size of feature maps (e.g., max pooling), helping to lower computation and retain important features.
  - Activation Layer – Introduces non-linearity (e.g., ReLU, Sigmoid) so the network can learn complex patterns.
  - Dropout Layer – Randomly disables some neurons during training to prevent overfitting.
  - Batch Normalization Layer – Normalizes outputs of a layer to stabilize and speed up training.
  - Output Layer – Produces the final prediction, often using activation functions like Softmax (for classification) or linear (for regression).
    
5. What does an epoch mean in model training?
  - In training a model, an epoch is defined as one complete pass of the entire training dataset through the neural network. In an epoch, the model goes through all the training samples once and learns from them. Usually, more than one epoch is required so that the model can improve its performance and reduce errors.
    
6. Compare the predicted label and actual label for the first test image.
  - The actual label of the first test image is the true class that it belongs to, while the predicted label is the class that the trained model predicts it to be. By comparing the two, we can determine whether the model has correctly identified the image or not.
    
7. What could be done to improve the model’s accuracy?
  - To improve the model’s accuracy,  use more training data, apply data augmentation to make the model see varied examples, adjust hyperparameters such as learning rate and batch size, modify the layers of the network to improve the learning of features, and implement dropout to avoid overfitting. Another technique you can implement is transfer learning using a pre-trained model if the size of your dataset is small.

    
 
.[This is the link to my Google Colab project:](https://colab.research.google.com/drive/1F-GPjtmh31Z0BPiHH74vh12ZHAjpVYoe#scrollTo=KjfT8vDxQQx6)


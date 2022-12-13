# Deep Learning (CSE 490G1) Final Project: American Sign Langugage Classification Using Neural Networks
**Group Members: Vivek Patel, Hayoung Jung**

## Abstract
To identify and classify American Sign Language (i.e. ASL) images into correct English alphabet characters, we employed convolutional neural networks in order to effectively execute the task quickly and accurately. The model was trained and tested on 87,000 ASL image dataset from Kaggle. We investigated and experimented with various model network architectures, image data augmentations, hyperparamters, and transfer learning with AlexNet in order to obtain the performance. Our best-performing model achieved an accuracy of 100% on our 29-image test suite of ASL characters while obtaining 98.7126% validation accuracy on the validation set. Our results suggest that 1) deep learning models can accurately classify ASL images, making them incredible tools for those who are beginning to learn American Sign Language, and 2) training simple models from scratch on a dataset can have a much better performance compared to trying to finetune a powerful, complicated network to a different task.  

## Problem Statement
American Sign Language (also known as ASL) is a visual language often used for communication by those who are deaf or hard of hearing. Live ASL interpreters are a facet of many public and private events as a means of increasing accessibility, but for individuals who are new to learning ASL, the interpreters can move too fast. Realtime ASL translation into a familiar language can be a good stepping stone for people in those circumstances, and a piece of that is characterizing individual ASL characters. We believe that neural networks can be incredibly useful tools for classifying each ASL character into english alphabet character accurately and quickly. In this project, we build a convolutional neural network (CNN) to classify a given image of an ASL sign into the correct English alphabet characters.

## Related Work
We used the ASL Alphabet dataset from Kaggle, which contains an 87,000 image training set and 29 image test suite of all the ASL character classes in the dataset. Each image was 200-by-200 pixels with 3 channels for RGB colors. The dataset was split into 29 classes, of which are 26 English alphabet characters and 3 other classes for NOTHING, SPACE, and DELETE. The dataset was designed for realtime applications and classifications -- the dataset can be found at this link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet. 

We also took inspiration for this idea based on an American Sign Language module within LING 200 course offered at the University of Washington (https://myplan.uw.edu/course/#/courses/LING200). Having learned about the significance of ASL for those with speech and hearing difficulties, we wanted to apply both our learning within this Deep Learning course and LING 200 by creating a classifer for ASL images for accurate English character translation.

Within our project, we also performed transfer learning with AlexNet, a previous state-of-the-art Convolutional Neural Network model designed for ImageNet Classification task, for finetuning on the ASL classification task. As a course reading, the AlexNet proved to be a powerful network for the ImageNet competition. We used the same model for this project for a transfer learning comparison -- we obtained the AlexNet through the PyTorch distribution. The article can be found here: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf.

We personally used PyTorch in order to facilitate the entire deep learning process, MatPlotLib for visualization and graphs, and Numpy for simple arithmetic with the arrays computation. Google Colab's GPU was used for training the models.

## Methodology
After organizing our dataset within Google Colab, we performed data augmentation on the images, specifically the RandomHorizontalFlip and the ColorJitter functions afforded by PyTorch. This data augumentation improved the accuracy of our model's prediction enormously. After such data augmentations, we performed an 85%-15% train and validation split on the 87,000 image training set to obtain 73,950 images for the train set and 13,050 images for the validation set. We then uploaded the data into the PyTorch's dataloader.

With the dataloader complete, we experimented with multiple network architectures and hyperparameters to extract features from the ASL images and classify them to the accurate English alphabet characters. We detail the network architectures, hyperparameter values, evaluations, and performance results that worked the best for our given tasks, including the transfer learning for AlexNet, in the next section, specifically the "Experiments/evaluation" section. For each of the network architectures, we show the training loss, validation loss, and validation accuracies graph over each epoch on the best hyperparameter value. At the end, we also show the test accuracy on our 29-test suite image set as well as each individual model predictions on the 29 test images. 

## Experiments/evaluation



## Results



## Examples



## Video


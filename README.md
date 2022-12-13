# Deep Learning (CSE 490G1) Final Project: American Sign Language Classification Using Neural Networks
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
After organizing our dataset within Google Colab, we performed data augmentation on the images, specifically the RandomHorizontalFlip and the ColorJitter functions afforded by PyTorch. Having tried multiple data augumentation combination, we found that the particular combination for data augmentation improved the accuracy of our model's prediction enormously. After such data augmentations, we performed an 85%-15% train and validation split on the 87,000 image training set to obtain 73,950 images for the train set and 13,050 images for the validation set. We then uploaded the data into the PyTorch's dataloader.

With the dataloader complete, we experimented with multiple network architectures and hyperparameters to extract features from the ASL images and classify them to the accurate English alphabet characters. We detail the network architectures, hyperparameter values, evaluations, and performance results that worked the best for our given tasks, including the transfer learning for AlexNet, in the next section, specifically the "Experiments/evaluation" section. For each of the network architectures, we show the training loss, validation loss, and validation accuracies graph over each epoch on the best hyperparameter value. At the end, we also show the test accuracy on our 29-test suite image set as well as each individual model predictions on the 29 test images. 

## Experiments/Evaluation
Since we are performing a multiclass classification, we calculated cross entropy loss with mean reduction across all the model architectures including the transfer learning for AlexNet in order to the train the models. Note that we used stochastic gradient descent (SGD) to train the model using the cross entropy loss for all the experiments. We evaluated the experiment and the model results by using the validation accuracy and test accuracy primarily -- we calculated the accuracy based on how often the model was able to correctly predict the image label accordingly based on the ASL image. We also used cross entropy loss on the validation set (averaged across batches) to evaluate the model and decide which hyperparameters to use, but accuracy was our main evaluation metric.

### Experiment 1: ASLv1 Model

Among many of the models we tried out, we found that this architecture (i.e. in various depth, width, pooling size, convolution size and placement within the net) gave us the fastest training time and incredibly accurate classification predictions via validation accuracy. After settling for this model architecture, we experimented with various hyperparameters and found the optimal hyperparameters that got us to the best performance in terms of validation accuracy. More on results and examples of the ASLv1 generations below.

Network Architecture: 
```
class ASLNet_Version1(nn.Module):
    def __init__(self):
        super(ASLNet_Version1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 2)
        self.maxpool1 = nn.MaxPool2d((3, 3))
        self.conv2 = nn.Conv2d(16, 256, 2)
        self.conv3 = nn.Conv2d(256, 256, 2)
        self.maxpool2 = nn.MaxPool2d((4, 4))
        self.fc1 = nn.Linear(256*256, 420)
        self.fc2 = nn.Linear(420, 29)


        self.accuracy = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
```

Best Performing Hyperparameter values for ASLv1:
```
BATCH_SIZE = 64
TEST_BATCH_SIZE = 10
EPOCHS = 8
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
```
### Experiment 2: ASLv2 Model

With wide success from Experiment 1, we tried out various derivations from ASLv1, including making the convolution layer wider and adding more convolutions -- we thought that this will allow us to distill and extract more features from the image pixels. In addition, we also tried a bigger fully connected layer and different kernel sizes for the maxpool. Surprisingly, having too complex network architecture actually hurt model performance while drastically lengthening the training time. In the end, ASLv2 has one extra convolution layer with wider fully connected layer but slightly smaller kernel size in the first maxpool layer. We then experimented with various hyperparameter values and found the set of hyperparameter values that allowed us to obtain the most optimal model performance possible. As shown later in the results and live demo video at the end, ASLv2 outperforms ASLv1 in terms of the test suite of 29 images by correctly predicting 100% of all the images correctly (over ASLv1's ~96% test accuracy, incorrectly predicting one of the 29 images) despite ASLv1's better validation accuracy. More on this on the results segment and the video.

Network Architecture: 
```
class ASLNet_Version2(nn.Module):
    def __init__(self):
        super(ASLNet_Version2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(135424, 512)
        self.fc2 = nn.Linear(512, 29)

        self.maxpool1 = nn.MaxPool2d((2, 2))
        self.maxpool2 = nn.MaxPool2d((4, 4))
        self.accuracy = None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

Best Performing Hyperparameter values for ASLv2:
```
BATCH_SIZE = 64
TEST_BATCH_SIZE = 10
EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
```

### Experiment 3: AlexNet Model (Transfer Learning/Finetuning to ASL Classification Task)

After experimenting with various CNN model architectures and training them from scratch, we decided to take advantage of an already pretrained and powerful network: AlexNet. As a major model that performed enormously well in the ImageNet competition, AlexNet seemed like an awesome and challenging opportunity for us to perform transform learning and finetune the model to our task for ASL classification. The Experiment 3 would be our transfer learning comparison compared to our first two experiments where we trained models from scratch using the dataset. 

For this experiment, given AlexNet's immense size and pretrained weights, we decided to freeze the weights for all layers before layer 19 in order to preserve the pretrained portion of the network. After that, we removed the last 2 pretrained layer within AlexNet, and added three layers: two layers being linear layers (i.e. Linear-20 and Linear-22) and one layer being ReLU activation layer (i.e. ReLU-21). For these added layers, specifically the two additional linear layers, we trained and finetuned those layers within AlexNet and changed the weights. 

Our hypothesis was that AlexNet, as a pretrained model for ImageNet classification, would have already been incredibly effective at extracting features from images, which is why we froze the weights for AlexNet in much of the layers in order to preserve this feature extraction ability. Surprisingly, as detailed in the results and the video, AlexNet with finetuning on the task actually performed the worst compared to ASLv1 and ASLv2, which were trained from scratch. More on this in the results segment and the video. 

Network architecture:
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 56, 56]          23,296
              ReLU-2           [-1, 64, 56, 56]               0
         MaxPool2d-3           [-1, 64, 27, 27]               0
            Conv2d-4          [-1, 192, 27, 27]         307,392
              ReLU-5          [-1, 192, 27, 27]               0
         MaxPool2d-6          [-1, 192, 13, 13]               0
            Conv2d-7          [-1, 384, 13, 13]         663,936
              ReLU-8          [-1, 384, 13, 13]               0
            Conv2d-9          [-1, 256, 13, 13]         884,992
             ReLU-10          [-1, 256, 13, 13]               0
           Conv2d-11          [-1, 256, 13, 13]         590,080
             ReLU-12          [-1, 256, 13, 13]               0
        MaxPool2d-13            [-1, 256, 6, 6]               0
AdaptiveAvgPool2d-14            [-1, 256, 6, 6]               0
          Dropout-15                 [-1, 9216]               0
           Linear-16                 [-1, 4096]      37,752,832
             ReLU-17                 [-1, 4096]               0
          Dropout-18                 [-1, 4096]               0
           Linear-19                 [-1, 4096]      16,781,312
           Linear-20                  [-1, 512]       2,097,664
             ReLU-21                  [-1, 512]               0
           Linear-22                   [-1, 29]          14,877
================================================================
```

We froze the weights for all layers before layer 19, and trained according to these hyperparameters (which gave us the best performance for the AlexNet):
```
BATCH_SIZE = 32
TEST_BATCH_SIZE = 10
EPOCHS = 10
LEARNING_RATE = 0.005
MOMENTUM = 0.6
WEIGHT_DECAY = 0.0005
```
## Results

### Result 1: ASLv1 Model
ASLv1 Performance:
* Epoch: 8
* Train Loss: 0.0533818
* Validation Loss: 0.028572216
* Validation Accuracy: 99.21%
* Test Accuracy: 96.428% (i.e. 27/28 examples correct)

![](./aslv1graphs.png)


### Result 2: ASLv2 Model
ASLv2 Performance:
* Epoch: 10
* Training Loss: 0.024703674171026816
* Validation Loss: 0.04181242462900606
* Validation Accuracy: 98.7126%
* Test: 100.0% (i.e. 28/28 examples correct)

![](./aslv2graphs.png)

### Result 3: AlexNet Model (Transfer Learning/Finetuning to ASL Classification Task)
Modified AlexNet Performance:

![](./alexnetgraphs_withbg.png)
* Epoch: 9
* Training Loss: 1.707148460999344
* Validation Loss: 0.04122457779687026
* Validation Accuracy: 54.24521072796935%
* Test Accuracy: 71.4285174% (i.e. 20/28 examples correct)

### Result Analysis and Takeaways
Overall, ASLv1 and ASLv2 performed great, with ASLv2 performing slightly better in test accuracy (on the 29 image test suite) while ASLv1 performing better on the validation accuracy. In training, we also noticed more consistent decrease in loss for ASLv2, which makes us think that the results seen in ASLv1 are slightly due to chance, as the training loss did fluctuate up and down in a slightly higher fashion. ASLv2 was able to overcome challenges in distinguishing and correctly predicting between two very subtly similar but different label images (more on the Live Demo video below). However, ASLv1 seems to also exhibit better generalizability compared to ASLv2 due to ASLv1's lower validation loss, higher train loss, and higher validation accuracy compared to that of ASLv2. More interestingly, AlexNet performed tangibly worse. If we were to do it again, we'd likely unfreeze all the weights within AlexNet so that the model can learn to extract features more effectively for ASL images -- however, this would come with a tradeoff with the training time taking very long as the features learned for the ImageNet competition are far more different from this task at hand. 

Training the models from scratch on custom architectures clearly yielded better results than using a powerful model with ImageNet classification weights and finetuning them for our task. We suspect this is due to physical differences between ASL characters being different features than those separating the classes in the ImageNet dataset. In addition, too complex model architecture may hurt the model's learning and performance, where too complicated CNN models trained from scratch actually performed worse and even with AlexNet, with complex model architecture, actually performed worse than its simpler counterpart ASLv1 and ASLv2. AlexNet may have been an overkill for this task with not enough training time to see the benefits with high accuracy.  

## Examples

Here, we detail the model's predictions for each of the 29 image test suite. The test suite contains each of the ASL characters and the NOTHING, SPACE, and DELETE classes. For each example, we show our experiment model's predictions against the labels for the 29 images.

### Examples 1: ASLv1 Model
![](./aslv1demo_withbg.png)

### Examples 2: ASLv2 Model
![](./aslv2demo_withbg.png)

### Examples 3: AlexNet Model (Transfer Learning/Finetuning to ASL Classification Task)
![](./alexnetdemo_withbg.png)

## Video

[Here](https://youtu.be/geiv9Ux8MT0) is a brief video presentation overviewing our project:

[![Here's a video of our presentation!](https://img.youtube.com/vi/geiv9Ux8MT0/0.jpg)](https://youtu.be/geiv9Ux8MT0)


And [here](https://youtu.be/pgPHhu32z9Q) is a live demo (extra credit!) of the networks making predictions:

[![Here's a video of our live demo!](https://img.youtube.com/vi/pgPHhu32z9Q/0.jpg)](https://youtu.be/pgPHhu32z9Q)

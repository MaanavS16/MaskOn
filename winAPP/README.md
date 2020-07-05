## Inspiration
With the current coronavirus outbreak, businesses struggle to handle the economic toll of increased operational costs through sanitation and reduced business. A large concern of a business owner is the risk of their store becoming a “Hot Spot” for coronavirus cases. Dozens of studies have proven that the simplest way to reduce the risk of Covid-19 is to simply: **Put your Mask On!** So we set out to create a software solution that works with existing security cameras to make sure that customers were wearing their masks.

## How it Works
The user experience starts with the **tkinter windows application** we have created. Individual frames are extracted from the live video feed from the webcam. These frames are reshaped, resized, and converted to a (256, 256, 3) byte array. The byte array is passed to the predict method of a **Convolutional Neural Network.** The results are then overlain with the video-feed and displayed.

## How we built it
The **core** of our project was the Convolutional Neural Network designed in python with an open source Machine Learning tool called **Tensorflow**. A Convolutional Neural Network is a special class of NN with the unique ability to recognized features of data that has a structure where adjacent values can be grouped together (like an image!). In our Convolutional Layers we passed a few dozen "filters" over the image to detect different features of the image and then passed those filtered images to a Deep Neural Network. After forward propagating the network will converge to a single value ranging between 0 and 1 indicating the predicted class and the confidence in that predication.

## The Data: 
In order to train our CNN we needed to have a significantly large classified set of images of people wearing face masks and not wearing them. We found a large open source face mask [dataset](https://github.com/prajnasb/observations/tree/master/experiementd) on Github. To further increase the generalizing abilities of the CNN we web-scrapped some tougher images to classify and added them to the dataset.

During training we used a technique called Image Augmentation, where we apply various transformations (zoom, skew, change saturation, etc.) are randomly applied to images in the training set; this allows for the neural network to learn image features even with significant distortion allowing it to better generalize to the variety of cameras and video capture devices.


## What's next for MaskOn
We would create a version of this project that works with tf.js to allow for a web experience. We could also create a REST API to allow for other services to detect if a customer is wearing a mask; this could be used for automated door that only allows mask wearing customers in.

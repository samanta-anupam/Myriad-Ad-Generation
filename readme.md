This project is guided by Professor Roy Shilkrot, Stony Brook University. 

In this project, we aim to develop a deep learning based system where we create a model that given a train dataset of image-ads, it can identify objects within the ad and then learn a latent representation of these types of layout. 

Using these vector latent representation, a lot of exciting things can be done on top of it.

One of them would be to generate new ads and other would be to assist in making the ad, wherein the user places some objects
and the deep learning based on its learning can position them and create a more appealing layout. This requires less knowledge of 
design at the user end meanwhile creating fantastic ads.

Our plan is to use a Object detection network on top of the images where it is to be trained. 
There are many capable networks such as YOLO, etc. We use the output of the YOLO network to create a new stacked image, 
where each layer maps to a particular class with binary values, indicating whether the pixel in the image is a part of 
an object identified by the YOLO network.
 
 
As YOLO has over 1000 classes, and it is unlikely that we are going to see all classes
within our dataset. Hence our initial plan would be to see the distribution of classes among all images, and compress the layers to 
a lesses number of layers. We can also club a few object classes to one layer. There are other approaches that can be thought
of here as an ingenious way to reduce the number of layers.

The last part of the project would be to train this layered image on an auto-encoder, which can learn the representation
of this object layout of the ad into a smaller encoded feature vector. 

Hence any arrangement of object layouts at the user end, would lead to a new feature vector, where we can pass the new 
image layout in our auto encoder, and the decoded output will be the new corrected position of the objects.     


This project is divided in the following files:

|File|Use|
|---|---|
|Myriad Ad Generation.ipynb|Main design generation notebook|
|Old Ad Generation.ipynb|Original Ad design generation notebook|
|Class Analysis.ipynb|Class Analysis generation|
|object-detection|[Google's tensorflow research model](https://github.com/tensorflow/models/tree/master/research/object_detection)|
|text-detection|[Text-detection feed forward model](https://github.com/argman/EAST)|

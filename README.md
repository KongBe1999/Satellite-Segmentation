# Satellite Segmentation
This is my first project about deep learning . Purpose of project is segmenting Satellite image but the additional important feature is computing the Square of several interested classes ( such as Building, Tree , ... ) .From that , it is better for land planing . I think it is great idea.

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## 1. Data
- Data is ISPRS ( International Society for Photogrammetry and Remote Sensing ) [here](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html) that provided for 2D Semantic Labeling Contest.
- Data have 38 patches( of the same size ) , each part of image is denoted in rows (left to right) and columns (north to south) of the map . Each image have 6000*6000 pixels.
- Images are [.tif](https://en.wikipedia.org/wiki/TIFF) files. 
![](https://user-images.githubusercontent.com/60954405/74927467-3ee7b480-540a-11ea-8b3f-25b85acc56c8.gif)
- There are 6 classes , include :
![](https://user-images.githubusercontent.com/60954405/74927921-18764900-540b-11ea-996d-ee4c35c33eeb.png)
## 2. Preprocessing 
- The segmentation images is RGB that have different color for each class. => Find the Gray value of each class.
- Original images have 6000*6000 (so big for input of model) , so i crop them into many 256*256 images.
- About the amount image , I split data into 3 part : Train, Validation and Test 
  + Train : 20 000 images (91,37%)
  + Validation : 1 000 images (4,56%)
  + Test : 888 images (4,07%)
- Create mask for predicted image through model .
## 3. Model 
![](https://user-images.githubusercontent.com/60954405/74928923-085f6900-540d-11ea-82e8-e0143b2eb9e9.png)
- I used Keras for convenient , fast .
- I trained model with epochs = 5, steps per epoch = 20 000, batch size = 5, optimizer = Adam (lr=0.0005). 
- Metric : "Accuracy", "F1-Score".
## 4. Result 
- The training process in Google Colab results learning curve , decribed below :
![](https://user-images.githubusercontent.com/60954405/74929901-0eeee000-540f-11ea-892d-4964042beb20.png)
- Evaluated on validation set

Loss | Accuracy | F1_score | Recall | Precision
---- | ----- | -------- | ------ | ---------
0.56400	| 0.84| 0.84321	| 0.85296	| 0.83369
- Evaluated on test set 

Loss | Accuracy | F1_score | Recall | Precision
---- | ----- | -------- | ------ | ---------
0.78046	| 0.79424	|0.79441|	0.80396|	0.78508

- Test on 1 image in test set at left , model predicted at middle and true label image at right :
![](https://user-images.githubusercontent.com/60954405/74930673-c0424580-5410-11ea-8ed3-e4aa123fb0fd.png)

![](https://user-images.githubusercontent.com/60954405/74930836-00092d00-5411-11ea-85df-2e33e4235a98.png)

## 5. References
In the process of implementing the project, from sketching ideas to completing models, I have consulted a number of resources including scientific articles, github, .... Here are some examples that i think they have important role in the implementation of this topic:
+ [The art of state for semantic segmentation 2019](https://heartbeat.fritz.ai/a-2019-guide-to-semantic-segmentation-ca8242f5a7fc)
+ [Explore models through articles with code](https://paperswithcode.com/task/semantic-segmentation/latest) 

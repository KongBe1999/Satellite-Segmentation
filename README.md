# Satellite Segmentation
This is my first project about deep learning . Purpose of project is segmenting Satellite image but the additional important feature is computing the Square of several interested classes ( such as Building, Tree , ... ) .From that , it is better for land planing . I think it is great idea.

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## 1. Data
- Data is ISPRS ( International Society for Photogrammetry and Remote Sensing ) [here](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html) that provided for 2D Semantic Labeling Contest.
- Data have 38 patches( of the same size ) , each part of image is denoted in rows (left to right) and columns (north to south) of the map . Each image have 6000*6000 pixels.
- Images are [.tif](https://en.wikipedia.org/wiki/TIFF) files. 

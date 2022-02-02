# Siamese Networks for point cloud quality assessment 

A sample code code from my internship work. 

The jupyter notebook src/Full-reference siamese network.ipynb shows the implementation of a convolutional siamese network trained and tested on ICIP 2020 dataset for full reference point cloud quality assessment.

The jupyter notebook src/No-reference siamese network.ipynb shows the implementation of a convolutional siamese network for no reference point cloud quality assessment trained on ModelNet dataset, fine tuned and tested on ICIP 2020 dataset.

The jupyter notebook src/New architecture based on convolutional siamese networks and PointNet.ipynb shows the implementation of a new architecture network for full reference point cloud quality assessment based on the previous convolutional siamese networks and Pointnet, trained on ModelNet dataset, fine tuned and tested on ICIP 2020 dataset.

The ICIP 2020 dataset can be dowloaded from https://drive.google.com/file/d/1MemLa255e0wrGXbWoVKDS5ghoqAmfWHw/view?usp=sharing

Some preprocessing functions used in this project (in src/utilities.py file) were taken from the Maurice Quach repository https://github.com/mauriceqch/2021_pc_perceptual_loss. These parts of code were highlighted by the comment : ### Quach    

# A Lightweight Dual Path Kolmogorov-Arnold Convolution Network for Medical Optical Image Segmentation
# skin lesion segmentation
We take skin disease segmentation as an example to introduce the use of our model.
# Data preparation
resize datasets ISIC2018 to 224*224 and saved them in npy format.
python data_preprocess.py
# Train and Test
Our method is easy to train and test, just need to run "train.py".
python train.py

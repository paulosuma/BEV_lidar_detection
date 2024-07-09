#!/bin/bash

#change directory to Datasets folder

# Download KITTI dataset files
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -O data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip -O data_object_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip -O data_object_calib.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip -O data_object_label_2.zip

# Extract the files
unzip data_object_image_2.zip
unzip data_object_label_2.zip
unzip data_object_calib.zip
unzip data_object_velodyne.zip

# Clean up ZIP files if needed
rm data_object_image_2.zip
rm data_object_label_2.zip
rm data_object_calib.zip
rm data_object_velodyne.zip


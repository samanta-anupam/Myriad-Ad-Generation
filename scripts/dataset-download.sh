#!/bin/bash
# bash script to download all files:
# get links for dataset to download from: https://people.cs.pitt.edu/~kovashka/ads/#image

# path to the dataset location
cd /gpfs/scratch/asamanta/dataset/train_images/

for i in `seq 0 10`; do 
	wget https://storage.googleapis.com/ads-dataset/subfolder-$i.zip;
	unzip subfolder-$i.zip;
	rm subfolder-$i.zip;
done

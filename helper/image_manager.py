import os

import numpy as np
from PIL import Image
from math import ceil
from tqdm import tqdm_notebook
from .config import *

# Image manager works as an iterator to fetch all images one by one.
class ImageManager:
    # initialize with number of partitions to pick from
    def __init__(self, start, end, overwrite=False):
        self.start = start
        self.end = end
        self.folder_path = IMAGES_PATH
        self.overwrite = overwrite
        self.od_output_dir = OBJECT_DET_OUTPUT
        self.td_output_dir = TEXT_DET_OUTPUT
        self.npy_data_dir = NUMPY_DATA
        self.index = 0
        self.cluster_file_name = CLUSTER_CLASSES_FILE_PATH
        self.num_mask, self.mapping_classes_index = self.get_mapping_classes_index()
        self.files = self.resized_images()
        self.totalFiles = len(self.files)

    # get next batch of size batch_size
    def next_batch(self, batch_size):
        start = self.index
        self.index += batch_size
        if self.index > self.totalFiles:
            self.index = 0
            file_order = np.arange(self.totalFiles)
            np.random.shuffle(file_order)
            self.files = self.files[file_order]
            start = 0
            self.index = batch_size
        return self.files[start : self.index]

    # get left right top bottom bound of an image
    def get_bounds(self, line, img_width, img_height, size):
        coord = [int(x) for x in line.split(',')]
        l = round(coord[0]*img_width/size[0])
        r = round(coord[2]*img_width/size[0])
        t = round(coord[1]*img_height/size[1])
        b = round(coord[-1]*img_height/size[1])
        return l,r,t,b

    # get and masks for all images in the dataset part i
    def get_images_and_masks(self, i):
        path = os.path.join(self.folder_path, str(i))
        file_names = os.listdir(path)
        masks = np.zeros((len(file_names), 100, 100, self.num_mask), np.uint8)
        delete_mask = []
        for idx, file_name in enumerate(tqdm_notebook(file_names,desc='Read:'+str(i))):
            name = os.path.basename(file_name).split('.')[0]
            td_output_file = os.path.join(self.td_output_dir, str(i), name)+'.txt'
            od_output_file = os.path.join(self.od_output_dir, str(i), name)+'.txt'
            if file_name.endswith('.jpg') and (os.path.isfile(td_output_file) 
                                               or os.path.isfile(od_output_file)):
                orig_img = Image.open(os.path.join(path, file_name))
                
                if os.path.isfile(td_output_file):
                    with open(td_output_file, 'rt') as file:
                        lines = file.read().splitlines()
                        for line in lines:
                            l,r,t,b = self.get_bounds(line, masks[idx].shape[0], masks[idx].shape[1], orig_img.size)
                            masks[idx,t:b,l:r,0] = 1
                            
                if os.path.isfile(od_output_file):       
                    with open(od_output_file, 'rt') as file:
                        lines = file.read().splitlines()
                        for line in lines:
                            try:
                                class_idx = self.mapping_classes_index[
                                    line.split(':')[0].split(',')[-1]]+1
                                coord = line.split(':')[1].split(',')
                                coord = [round(float(x)*masks[idx].shape[(j+1)%2]) for j,x in enumerate(coord)]
                                masks[idx,coord[0]:coord[2],coord[1]:coord[3],
                                      class_idx] = 1
                            except KeyError:
                                pass
            else:
                delete_mask.append(idx)
                
        np.delete(masks, delete_mask)
        if not os.path.exists(self.npy_data_dir):
            os.makedirs(self.npy_data_dir)
        np.save(os.path.join(self.npy_data_dir, str(i)), masks)
        return masks
    
    # resize all images of the dataset to fit size specified during init.
    def resized_images(self):
        final_masks = None
        for i in tqdm_notebook(range(self.start, self.end), desc='1st loop'):
            if not os.path.isfile(os.path.join(self.npy_data_dir, str(i))+'.npy') or self.overwrite:
                masks = self.get_images_and_masks(i)
            else:
                masks = np.load(os.path.join(self.npy_data_dir, str(i))+'.npy')
            if final_masks is None:
                final_masks = masks
            else:
                final_masks = np.append(final_masks,masks,axis=0)
        return final_masks
    
    # get classes and their index.
    def get_mapping_classes_index(self):
        clustered_classes = []
        mapping_classes_index = dict()
        clustered_classes_count = []
        file_name = self.cluster_file_name
        
        with open(file_name, 'rt') as file:
            lines = file.read().splitlines()
            idx = -1
            for line in lines:
                words = line.split('\t')
                if not line.startswith('\t'):
                    clustered_classes.append(set())
                    clustered_classes_count.append(0)
                    idx += 1
                class_name = line.strip().split('\t')[0]
                mapping_classes_index[class_name] = idx
                clustered_classes_count[idx] += int(line.strip().split('\t')[-1])
                clustered_classes[idx].add(class_name)

        class_with_count = list(zip(clustered_classes, clustered_classes_count))
        class_with_count.sort(key=lambda x: x[1], reverse=True)


        filtered_classes = [x[0] for i,x in enumerate(class_with_count) if x[1]>500]
        filtered_classes_count = [x[1] for i,x in enumerate(class_with_count) if x[1]>500]
        mapping_filtered_classes = dict()
        for i,classes in enumerate(filtered_classes):
            for c in classes:
                mapping_filtered_classes[c] = i

        clustered_classes = filtered_classes
        clustered_classes_count = filtered_classes_count

        mapping_classes_index = mapping_filtered_classes

        NUM_CLASSES = len(clustered_classes)
        NUM_BYTES_FOR_MASK = NUM_CLASSES+1
        return NUM_BYTES_FOR_MASK, mapping_classes_index
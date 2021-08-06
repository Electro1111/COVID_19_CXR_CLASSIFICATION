import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from robbie_lib import ROBBIE
from skimage.color import label2rgb
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from heapq import heappop, heappush, heapify
import heapq
from helper_functions import *
import time

FOLDER_PATH = "../../segmentation_models/null"
path_to_data="../../datasets/classification/data"
path_to_data_lungs_removed="../../datasets/classification/data_lungs_removed"
path_to_data_lungs_boxed_out="../../datasets/classification/data_lungs_boxed_out"
path_to_data_lungs_isolated="../../datasets/classification/data_lungs_isolated"
path_to_data_lungs_framed="../../datasets/classification/data_lungs_framed"

train_list_path="../COVID-Net/train_COVIDx5.txt"
test_list_path="../COVID-Net/test_COVIDx5.txt"

show = False

# LOAD NETWORK

print("initializing unet model")
network= ROBBIE(D = 5, 
                W = 6, 
                LR = .0001,
                model_name = "null",
                data_handle = '_mask', 
                load = "../../segmentation_models/UNET_NIH_A_MODEL/NIH_A_MODEL_BEST.pth",
                folder = FOLDER_PATH
               )
import shutil
shutil.rmtree(FOLDER_PATH) #remove the save path created by ROBBIE because it is not needed



#first verify that all images needed for training and testing are in the data folder
print("Check and make sure all images in COVIDx5 are in: ", path_to_data)

train_file_text = open(train_list_path,'r').readlines()

test_file_text = open(test_list_path,'r').readlines()

train_files = [line.split()[1] for line in train_file_text]

test_files = [line.split()[1] for line in test_file_text]


#now load names of files actually in the train folder:

train_files_folder = os.listdir(os.path.join(path_to_data,'train'))
test_files_folder = os.listdir(os.path.join(path_to_data,'test'))

print(len(train_files_folder)+len(test_files_folder), " files to parse")

if len(set(train_files).difference(set(train_files_folder)))==0:
    print("training set checks out")
else:
    print("ERROR, not all train files are in data path... exiting")
    exit()
    
if len(set(test_files).difference(set(test_files_folder)))==0:
    print("test set checks out")
else:
    print("ERROR, not all test files are in data path... exiting")
    exit()
    
    
for separation in ['train', 'test']:
 
    for count,file in enumerate(os.listdir(os.path.join(path_to_data,separation))):
        
        t0=time.time()
    
        lungs_inpath=os.path.join(*[path_to_data,separation,file])
        lungs_removed_outpath = os.path.join(*[path_to_data_lungs_removed,separation,file])
        lungs_boxed_out_outpath = os.path.join(*[path_to_data_lungs_boxed_out,separation,file])
        lungs_isolated_outpath = os.path.join(*[path_to_data_lungs_isolated,separation,file])
        lungs_framed_outpath = os.path.join(*[path_to_data_lungs_framed,separation,file])
        
        '''print(lungs_inpath)
        print(lungs_removed_outpath)
        print(lungs_boxed_out_outpath)
        print(lungs_isolated_outpath)
        print(lungs_framed_outpath)'''
        
        if os.path.exists(lungs_inpath):
            original_image = cv2.imread(lungs_inpath,0)
            
        
            square_image,padding = squarify(original_image,0)

            small_square_image = cv2.resize(square_image, dsize=(512,512))

            small_square_image_mask = network.infer(small_square_image) #run model on image

            square_image_mask = cv2.resize(small_square_image_mask.astype('uint8'), dsize = (square_image.shape[1],square_image.shape[0]))

            original_image_mask = remove_padding(square_image_mask,padding)

            segments = label((original_image_mask>0).astype(int))
            
            overlay = label2rgb(segments,original_image,bg_label=0)
            
            if show==True:
                plt.figure(figsize = (20,4))
                plt.suptitle(file)

                plt.subplot(1,7,1)
                plt.title("original")
                plt.imshow(original_image)

                plt.subplot(1,7,2)
                plt.title("square")
                plt.imshow(square_image)

                plt.subplot(1,7,3)
                plt.title("small square")
                plt.imshow(small_square_image)

                plt.subplot(1,7,4)
                plt.title("small square mask")
                plt.imshow(small_square_image_mask)

                plt.subplot(1,7,5)
                plt.title("square mask")
                plt.imshow(square_image_mask)

                plt.subplot(1,7,6)
                plt.title("original mask")
                plt.imshow(original_image_mask)

                plt.subplot(1,7,7)
                plt.title("overlay")
                plt.imshow(overlay)
                
                plt.savefig(os.path.join('./step1', file+'.png'))
                    
            
            
            h = []
            for blob in regionprops(segments):
                heapq.heappush(h,(-blob.area,blob.label))
                
                
            

            if len(h)>1:
                lung1_label=heapq.heappop(h)[1]
                lung2_label=heapq.heappop(h)[1]

                lung1_mask=segments==lung1_label
                lung2_mask=segments==lung2_label

                lung1_box = np.zeros(original_image.shape)
                lung2_box = np.zeros(original_image.shape)

                minr, minc, maxr, maxc = regionprops(lung1_mask.astype(int))[0].bbox
                lung1_box[minr:maxr,minc:maxc]=1

                minr, minc, maxr, maxc = regionprops(lung2_mask.astype(int))[0].bbox
                lung2_box[minr:maxr,minc:maxc]=1

                both_lungs = (lung1_mask+lung2_mask)>0
                #both_lungs = (lung1_mask+lung2_mask)>0
                both_boxes = (lung1_box+lung2_box)>0
                #both_boxes = (lung1_box+lung2_box)>0
            else:
                lung1_label=heapq.heappop(h)[1]
                lung1_mask=segments==lung1_label
                lung1_box = np.zeros(original_image.shape)
                lung2_box = 0
                lung2_mask=0
                minr, minc, maxr, maxc = regionprops(lung1_mask.astype(int))[0].bbox
                lung1_box[minr:maxr,minc:maxc]=1
                both_lungs = (lung1_mask)>0
                #both_lungs = (lung1_mask)>0
                both_boxes = (lung1_box)>0
                #both_boxes = (lung1_box)>0
            
            
            
            
            #lungs_removed_image = original_image*(1-(lung1_mask+lung2_mask))
            lungs_removed_image = np.copy(original_image)
            lungs_removed_image[both_lungs]=0
            
            lungs_isolated_image = np.copy(original_image)
            lungs_isolated_image[~both_lungs]=0
            
            #lungs_boxed_out_image = original_image*(1-(lung1_box+lung2_box))
            lungs_boxed_out_image = np.copy(original_image)
            lungs_boxed_out_image[both_boxes]=0
            
            lungs_framed_image = np.copy(original_image)
            lungs_framed_image[~both_boxes]=0
            
            
            
            if show==True:
                viz = mark_boundaries(original_image,(lung1_mask+lung2_mask+lung2_box+lung1_box).astype(int), color = (1,1,0),outline_color = (1,0,1))




                viz = label2rgb(lung1_mask+lung2_mask+lung2_box+lung1_box, original_image, bg_label=0)
                plt.figure(figsize = (20,10))
                plt.subplot(1,5,1)
                plt.imshow(original_image)
                plt.subplot(1,5,2)
                plt.imshow(lungs_removed_image)
                plt.subplot(1,5,3)
                plt.imshow(lungs_isolated_image)
                plt.subplot(1,5,4)
                plt.imshow(lungs_boxed_out_image)
                plt.subplot(1,5,5)
                plt.imshow(lungs_framed_image)
                plt.savefig(os.path.join('./step2', file+'.png'))
            
            cv2.imwrite(lungs_removed_outpath, lungs_removed_image)
            cv2.imwrite(lungs_isolated_outpath, lungs_isolated_image)
            cv2.imwrite(lungs_boxed_out_outpath, lungs_boxed_out_image)
            cv2.imwrite(lungs_framed_outpath, lungs_framed_image)
            
            
            t1=time.time()
            print(lungs_removed_outpath)
            print("computed in: ",(t1-t0))
            
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
print("...TASK COMPLETED...")
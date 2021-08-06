import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage.measure import label
from skimage.color import label2rgb
print('creating lung segmentation dataset')

#path to cohen segmentation masks and data
cohen_ann="../data_sources/covid-chestxray-dataset/annotations/lungVAE-masks"
cohen_img="../data_sources/covid-chestxray-dataset/images"

#path to XSLOR segmentation training data masks and CXR images
XLSor = "../data_sources/XLSor_data/data/Augmentation"

#path to NIH segmentation test set
NIH_img = "../data_sources/XLSor_data/data/NIH/images"
NIH_ann = "../data_sources/XLSor_data/data/NIH/masks"

#path to test data folders
cohen_test_save_dir = "../datasets/segmentation/cohen_test/"
NIH_test_save_dir = "../datasets/segmentation/NIH_test/"

#path to train data folder
train_save_dir = "../datasets/segmentation/train/"

#first lets create the test data

#annotation files:
ann_files = os.listdir(cohen_ann)
img_files = os.listdir(cohen_img)

from math import ceil, floor
def squarify(M,val):
    
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(ceil((a-b)/2),floor((a-b)/2)))
    else:
        padding=((ceil((b-a)/2),floor((b-a)/2)),(0,0))
        
    out=np.pad(M,padding,mode='constant',constant_values=val)
    
    if (out.shape[0]!=out.shape[1]):
        print('error when squraifying')
        exit()
   
    return np.pad(M,padding,mode='constant',constant_values=val)

count = 0
pairs={}
for img_file in img_files:
    for ann_file in ann_files:
        key_check = ann_file.split("_mask")[0]
        value_check = '.'.join(img_file.split('.')[:-1])
       
        if key_check==value_check:
            if ann_file in pairs.keys():
                print("key error, already exists")
                '''print(ann_file)
                print(pairs[ann_file])
                print(img_file)'''
            else:
                pairs[ann_file]=img_file
                '''print(ann_file)
                print(img_file)'''
            
            count+=1
            
print(len(pairs))
print(count)
show = False

from skimage.measure import label
from skimage.color import label2rgb

count = 0

for ann_file,img_file in pairs.items():
    
    
    
    ann = cv2.imread(os.path.join(cohen_ann,ann_file),0)
    img = cv2.imread(os.path.join(cohen_img,img_file),0)
    
    img = squarify(img,0)
    ann = squarify(ann,0)
    img = cv2.resize(img,(512,512))
    ann = (cv2.resize(ann,(512,512))>(255/2)).astype(int)*255
    
    new_name= "IMAGE_"+str(count)
    count+=1
    
    
    '''cv2.imwrite(save_dir+new_name+"_mask.png",ann)
    cv2.imwrite(save_dir+new_name+".png",img)'''
    
    ann_out_path = os.path.join(cohen_test_save_dir,new_name+"_mask.png")
    img_out_path = os.path.join(cohen_test_save_dir,new_name+".png")
    
    
    
    
    
    
    cv2.imwrite(ann_out_path,ann)
    cv2.imwrite(img_out_path,img)
    
    disp = label2rgb(ann,img, bg_label = 0)
  
    
    if show == True:
        plt.figure(figsize = (5,5))
        print(ann_file)
        print(img_file)
        print('saving to..')
        print(ann_out_path)
        print(img_out_path)
        plt.imshow(disp)
        plt.axis('off')
        plt.show()
    
    
        
   
   
for img_file in os.listdir(NIH_img):
    ann_file = img_file.split('.')[0]+'_mask.png'
    
    img = cv2.imread(os.path.join(NIH_img,img_file))
    ann = cv2.imread(os.path.join(NIH_ann,ann_file))
    
    if show==True:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.subplot(1,2,2)
        plt.imshow(ann)
        plt.show()
    
    ann_outpath = os.path.join(NIH_test_save_dir, ann_file)
    img_outpath = os.path.join(NIH_test_save_dir, img_file)
    
    cv2.imwrite(ann_outpath, ann)
    cv2.imwrite(img_outpath, img)

#now lets do the training data
from skimage.segmentation import mark_boundaries
os.listdir(XLSor)

ann_files = []
img_files = []

count = 0
for file in os.listdir(XLSor):
    if "mask" in file:
        
        ann_path = os.path.join(XLSor,file)
        ann=cv2.imread(ann_path,0)
        #ann=cv2.imread("XLSor_data/data/Augmentation/"+file,0)
        tag=file.split("mask")[0]
        
        for i in range(6):
            
           
                
            #img_file = tag+str(i)+".png"
            
            img_path=os.path.join(XLSor, tag+str(i)+".png")
            #img=cv2.imread("XLSor_data/data/Augmentation/"+img_file,0)
            img=cv2.imread(img_path,0)
            
            
            
            
            new_tag = "IMAGE_"+str(count)
            count+=1
            img_out_path = os.path.join(train_save_dir, new_tag+'.png')
            ann_out_path = os.path.join(train_save_dir, new_tag+'_mask.png')
            
            cv2.imwrite(img_out_path,img)
            cv2.imwrite(ann_out_path,ann)
            
            disp = label2rgb(ann,img, bg_label = 0)
            #disp = mark_boundaries(img,ann)
            if show==True:
                print(img_path)
                print(ann_path)
                print(img_out_path)
                print(ann_out_path)

                plt.figure(figsize = (5,5))
                plt.imshow(disp)
                plt.axis('off')
                plt.show()
     
            
            
            
            
            
        
print('done')


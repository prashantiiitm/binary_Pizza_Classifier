# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:28:46 2020

@author: prashant.pandey
"""

from skimage import io
from skimage.transform import resize
import os
import glob
import pandas as pd,numpy as np
import matplotlib.pyplot as plt

######## Set up the address of the current running script
try:
    filepath = os.path.abspath(os.path.dirname(__file__))
except:
    filepath = "D:/Project Files/Personal/Pizza_Classifier/Codes"
os.chdir(filepath)

##########################################################

folder_names = ("pizza","non_food","natural_images","food")

img_pixels = pd.DataFrame()

for i in folder_names:
    img_file_names =  glob.glob("..\\Images\\raw\\"+i+"\\**\\*.jpg",recursive = True)
    
    for img_id in img_file_names:
            img = io.imread(img_id)
            img_details = {"img_name" : [img_id], "folder" : [i], "len" : img.shape[0], "bth" : img.shape[1]  }
            img_pixels = img_pixels.append(pd.DataFrame(img_details))
    

#unq = img_pixels.groupby(['folder']).count()    
#    
#avg_pixels =  img_pixels.groupby(['folder']).mean()
#
#min_pixels =  img_pixels.groupby(['folder']).min()
#
#max_pixels =  img_pixels.groupby(['folder']).max()
            
            
img_small_avg = img_pixels[img_pixels['len'] < 600].groupby('folder').mean()


################## transform and label images to a same size 

img_pixels = pd.DataFrame()

piz_ctr = 1
npiz_ctr = 1

for i in folder_names:
    img_file_names =  glob.glob("..\\Images\\raw\\"+i+"\\**\\*.jpg",recursive = True)
    
    for img_id in img_file_names:
            
        
            img = io.imread(img_id)
            resized_image = resize(img,(200,200),anti_aliasing = True)
            if len(resized_image.shape) == 3:
                resized_image = resized_image[:,:,:3]
            
            if i == "pizza" : 
                filename = "..\\Images\\processed\\piz_" + str(piz_ctr) + ".jpg"
                piz_ctr = piz_ctr + 1
            else :
                filename = "..\\Images\\processed\\npiz_" + str(npiz_ctr) + ".jpg"
                npiz_ctr = npiz_ctr + 1             
            io.imsave(filename,resized_image)
                


npiz_img = int(11900/5)
piz_img = int(9214/5)

img_file_names =  glob.glob("..\\Images\\processed\\**\\*.jpg",recursive = True)

from sklearn.utils import shuffle
img_file_names = shuffle(img_file_names)

import shutil
piz_ctr = 1
npiz_ctr = 1

for img_name in img_file_names:
    if img_name.find("\\npiz") > 0:
        if(npiz_ctr < npiz_img) :
            str_pos = img_name.find("\\npiz") 
            file_name = img_name[str_pos+1:]
            shutil.move(img_name, "..\\Images\\processed\\testing\\" + file_name)
            npiz_ctr = npiz_ctr + 1
    elif img_name.find("\\piz") > 0:
        if(piz_ctr < piz_img) :
            str_pos = img_name.find("\\piz") 
            file_name = img_name[str_pos+1:]
            shutil.move(img_name, "..\\Images\\processed\\testing\\" + file_name)
            piz_ctr = piz_ctr + 1






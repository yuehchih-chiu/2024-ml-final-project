"""
這段程式碼的目的是將一個包含手寫數字和數學符號的影像資料集轉換為可以用於訓練機器學習模型的格式，並將其保存到 CSV 檔案中。
"""



import numpy as np
import pandas as pd
import cv2
import os
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
index_dir={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'+':10,'-':11,'times':12}
def get_index(directory):
    return index_dir[directory]

def load_images(folder):
    train_data = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE) 
        #img = ~img
        if img is not None:
            img = ~img
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0]) 
            m = 0
            for c in cnt:
                x, y, w, h = cv2.boundingRect(c)
                m = max(w*h, m)
                if m == w*h:
                    x_max,y_max,w_max,h_max=x,y,w,h
            im_crop = img[y_max:y_max+h_max+10, x_max:x_max+w_max+10] 
            im_resize = cv2.resize(im_crop, (28, 28)) 
            im_resize = np.reshape(im_resize, (784, 1)) 
            train_data.append(im_resize)
    return train_data
dataset_dir = './dataset/'
directory_list = listdir(dataset_dir)
#print(directory_list)
first = True
data = []
print('Imporitng...')
for directory in directory_list:
        print(directory)
        if first:
            first = False
            data = load_images(dataset_dir + directory)
            for i in range(0, len(data)):
                data[i] = np.append(data[i], [str(get_index(directory))])
            continue

        auxillary_data = load_images(dataset_dir + directory)
        for i in range(0, len(auxillary_data)):
            auxillary_data[i] = np.append(auxillary_data[i], [str(get_index(directory))])
        data = np.concatenate((data, auxillary_data))

df=pd.DataFrame(data,index=None)
df.to_csv('model/train_data.csv',index=False)

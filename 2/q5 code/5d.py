import numpy as np
import cv2
import matplotlib.pyplot as plt
import collections

def get_dct_feature(block):
    """make zigzag dct feature for a 8*8 normal block

    Args:
        block (numpy array 8*8)

    """    
    zigzag = np.array([[0  , 1 ,  5 ,  6 , 14 , 15 , 27 , 28]

                    ,[2,   4 ,  7 , 13,  16 , 26,  29,  42]

                    ,[3,   8,  12 , 17 , 25 , 30 , 41 , 43]

                    ,[9 , 11 , 18 , 24 , 31 , 40 , 44 , 53]

                    ,[10 , 19 , 23 , 32 , 39 , 45 , 52,  54]

                    ,[20 , 22  ,33,  38  ,46,  51 , 55 , 60]

                    ,[21  ,34 , 37 , 47  ,50 , 56 , 59 , 61]

                    ,[35  ,36 , 48 , 49,  57,  58 , 62 , 63]])
    
    zigzag_vec = zigzag.reshape(64,)
    dct = cv2.dct(block) # DCT transformer
    dct_vec = np.abs(dct.reshape(64,))
    sec_larg_index = np.argpartition(dct_vec,-2)[-2] #finding the second largest abs value of DCTs
    return zigzag_vec[sec_larg_index]
    
#loading training images
img = cv2.imread('leo1.png',0)
mask_img = cv2.imread('leo1_mask.png',0)
#normalize image for dct
noraml_img = img/255
features = np.zeros((img.shape[0]//8 , img.shape[1]//8))

prior_messi = np.sum(mask_img/255)/(mask_img.shape[0]*mask_img.shape[1])
prior_field = 1 - prior_messi
#calculating feature for each block
for i in range(img.shape[0]//8):
    for j in range(img.shape[1]//8):
        block = noraml_img[8*i:8*(i+1),8*j:8*(j+1)]
        features[i,j] = get_dct_feature(block)
     
features_vec = features.reshape((img.shape[0]//8 )* (img.shape[1]//8),)
mask_features = np.zeros((img.shape[0]//8 , img.shape[1]//8))

#deciding if each block in training image belongs to feild or messi
for i in range(mask_img.shape[0]//8):
    for j in range(mask_img.shape[1]//8):
        block = mask_img[8*i:8*(i+1),8*j:8*(j+1)]/255
        if np.sum(block) > 32:
            mask_features[i,j] = 1
        else:
            mask_features[i,j] = 0
    
mask_features_vec = mask_features.reshape((mask_img.shape[0]//8) * (mask_img.shape[1]//8),)

post_messi = []
post_field = []
#saving each feature to its class
for i in range((mask_img.shape[0]//8 )* (mask_img.shape[1]//8)):
    if mask_features_vec[i] == 1:
        post_messi.append(features_vec[i])
    else:
        post_field.append(features_vec[i])
      

#calculating likelihoods
counter_messi = collections.Counter(post_messi)

counter_field = collections.Counter(post_field)

for i in counter_field.keys():
    counter_field[i]/= len(post_field)
    
for i in counter_messi.keys():
    counter_messi[i]/= len(post_messi)
    
for i in range(64):
    if i not in counter_field.keys():
        counter_field[i] = 0
        
    if i not in counter_messi.keys():
        counter_messi[i] = 0
 
#loading test image
test_img = cv2.imread('leo2.png',0)
normal_test = test_img / 255

test_features = np.zeros((test_img.shape[0]//8 , test_img.shape[1]//8))
# calculating the X feature for each block
for i in range(test_img.shape[0]//8):
    for j in range(test_img.shape[1]//8):
        block = normal_test[8*i:8*(i+1),8*j:8*(j+1)]
        test_features[i,j] =get_dct_feature(block)
        x = test_features[i,j]
        if counter_messi[x]*prior_messi>counter_field[x]*prior_field:
            #test_img[8*i:8*(i+1),8*j:8*(j+1)]=np.ones((8,8))*255
            pass
        else:
            test_img[8*i:8*(i+1),8*j:8*(j+1)]=np.zeros((8,8))
            
cv2.imwrite('result.png',test_img)
import numpy as np
import cv2


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
    

 
#loading test image
test_img = cv2.imread('leo2.png',0)
normal_test = test_img / 255

test_features = np.zeros((test_img.shape[0]//8 , test_img.shape[1]//8))
# calculating the X feature for each block
for i in range(test_img.shape[0]//8):
    for j in range(test_img.shape[1]//8):
        block = normal_test[8*i:8*(i+1),8*j:8*(j+1)]
        test_features[i,j] =get_dct_feature(block)
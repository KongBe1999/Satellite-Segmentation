from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2

COLOR_DICT = {'Grass' : [0,0,255] , 'Building' :[0,255,255], 'Tree' :[0,255,0] , 'Car':[255,255,0] , 'Sand':[255,0,0],
                        'Road':[255,255,255], 'Unlabeled':[0,0,0] }
Grass =[0,0,255]
Building=[0,255,255]
Tree=[0,255,0]
Car=[255,255,0]
Sand=[255,0,0]
Road=[255,255,255]
COLOR_DICT = np.array([Grass, Building, Tree, Car, Sand,Road])
label=[29,179,149,225,76,255]

def testGenerator(test_path,num_image = 10,target_size = (256,256),as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = gray / 255
        gray = np.reshape(gray,(1,)+gray.shape+(1,))
        yield gray
def labelVisualize(num_class,color_dict,img):
    for i in range(256):
      for j in range(256):
        tmp = img[i][j]
        new_label = [int(k==np.max(tmp)) for k in tmp]
        img[i][j] = np.array(new_label)
    mask_RGB = np.zeros((img.shape[0],img.shape[1]) + (3,))
    mask_gray = np.sum(img*label, axis=2)
    for i in range(num_class):
      idx=np.where(mask_gray==label[i])
      mask_RGB[idx]=COLOR_DICT[i]
    return mask_RGB

def saveResult(save_path,npyfile,flag_multi_class = True,num_class = 6):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item)#img segmentation
        #cái này là kết quả cuối cùng
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)#save result

def saveResult_true(save_path,test_path,num_image = 100,flag_multi_class = True,num_class = 6): # lưu ảnh label đúng vào path trên
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d_Label.tif"%(i)))
        io.imsave(os.path.join(save_path,"%d_true.png"%(i)),img)#save ảnh label đúng

#### Addtion
# List all classes of predict image
def classes(mask_predict): # mask_predict has shape (256,256,6) , because it's go through the transfer_mask function to transfer one-hot form
  mask_predict = mask_predict.flatten()
  classes = []
  for i in range(0,len(mask_predict),6):
    if mask_predict[i]==1 :
      classes.append('Building')
    if mask_predict[i+1]==1 :
      classes.append('Grass')
    if mask_predict[i+2]==1 :
      classes.append('Tree')
    if mask_predict[i+3]==1 :
      classes.append('Car')
    if mask_predict[i+4]==1 :
      classes.append('Sand')
    if mask_predict[i+5]==1 :
      classes.append('Road')
  return classes    # classes là list gồm các lớp được liệt kê qua 1 lần chạy mask_predict (1 ảnh)

# Tính toán diện tích lớp được quan tâm
# Lý thuyết : Pixel là phần tử nhỏ nhất cấu thành nên ảnh và video trong các thiết bị điện tử
# nên px có mối quan hệ với các đại lượng đo chiều dài khác . Nhưng px không mang một mối quan hệ xác đinh
# nào cả . Nên ta cần chuẩn hóa như sau. Với bài toán đang làm thì thông số người ta đưa ra 1 ảnh Ortho(RGB)
# có kích thước 6k*6k*3 , trong đó 1px = 0.012 mm . => cần chuẩn hóa về mối quan hệ này.
# Tính toán S trên ảnh phân vùng :
# S_Building = Tổng số pixel mà phân vùng Building đó chiếm = n * 1px * 1px = n * 0.012^2 (mm) trong đó n là số px của phần vùng BUilding
# Ngoài ra người ta còn cung cấp với mức chuẩn hóa đã cho (1px = 0.012mm) thì Ty le anh chup ve tinh 1 : 8 000 000
# ==> S_Building_real = S_Building * 8 000 000 = ??? (mm^2)
def Square(array_mask):  # shape array_mask = (256,256,6)
  S, px2 = [], 0.012*0.012
  #array_mask = array_mask.reshape(array_mask, (256,256,6))
  S_Building = np.sum(array_mask[...,0])*8000000*px2   # (mm^2)
  S_Road = np.sum(array_mask[...,5])*8000000*px2       # (mm^2)
  S_Tree = np.sum(array_mask[...,2])*8000000*px2       # (mm^2)
  return [S_Building, S_Road, S_Tree]

def saveText(save_path, npyfile):   #### Lưu text có thông tin về số lớp của ảnh seg và diện tích từng vùng
  for n,img in enumerate(npyfile):
    img = np.resize(img, (256,256,6))
    for i in range(256):
      for j in range(256):
        tmp = img[i][j]
        new_label = [int(k==np.max(tmp)) for k in tmp]
        img[i][j] = np.array(new_label)
    S = Square(img)
    dict ={i for i in classes(img)}
    f = open(f"C:\\Users\\Kong Be\\Desktop\\unet_segmentation\Predict\\{n}_descibre.txt", "w+")
    f.write(f"{n}th image \nImage have : {dict} \nAbout image:S_Building={S[0]} mm^2, S_Road={S[1]} mm^2, S_Tree={S[2]} mm^2")
    f.close()



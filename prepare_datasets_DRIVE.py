import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from pre_processing import my_PreProc


def write_hdf5(arr,outfile):
  with h5py.File(outfile,"w") as f:
    f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
original_imgs_train = "/Users/wangxuelin/Dropbox/Huijing_XueLin/Imaging_Data/train_data/image/"
groundTruth_imgs_train = "/Users/wangxuelin/Dropbox/Huijing_XueLin/Imaging_Data/train_data/ground truth/"
#test
original_imgs_test = "/Users/wangxuelin/Dropbox/Huijing_XueLin/Imaging_Data/test_data/"
#---------------------------------------------------------------------------------------------

Nimgs = 1
test_Nimgs=6
channels = 3
N_div=6
height = 3000
width = 3000
height_new=500
width_new=500
dataset_path = "/Users/wangxuelin/Dropbox/Huijing_XueLin/Imaging_Data/training_testing/"

def get_datasets(imgs_dir,groundTruth_dir,train_test="null"):
    imgs = np.empty((Nimgs*N_div**2,height_new,width_new,channels))
    tmp_imgs = np.empty((Nimgs,height,width,channels))
    groundTruth = np.empty((Nimgs*N_div**2,height_new,width_new))
    tmp_gts = np.empty((Nimgs,height,width))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            #corresponding ground truth
            groundTruth_name = files[i][0:6] + "groundtruth.tif"
            print ("ground truth name: " + groundTruth_name)
            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            img=np.asarray(img)
            #img= img[500:3500,400:3400]
            g_truth=np.asarray(g_truth)
            #g_truth= g_truth[500:3500,400:3400]
            tmp_imgs[i]=img
            tmp_gts[i]=g_truth
    
    #reshaping for my standard tensors        
    tmp_imgs = np.transpose(tmp_imgs,(0,3,1,2))
    tmp_gts = np.reshape(tmp_gts,(Nimgs,1,height,width))
    tmp_imgs=my_PreProc(tmp_imgs)
    
    
    
    imgs,groundTruth = get_training_division(tmp_imgs,tmp_gts,N_div)


    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    assert(np.max(groundTruth)==255)
    assert(np.min(groundTruth)==0 )
    print ("ground truth is correctly withih pixel value range 0-255 (black-white)")
    assert(imgs.shape == (Nimgs*N_div**2,channels,height_new,width_new))
    assert(groundTruth.shape == (Nimgs*N_div**2,1,height_new,width_new))
    return imgs, groundTruth


def get_test_datasets(imgs_dir,train_test="null"):
    imgs = np.empty((test_Nimgs*N_div**2,height_new,width_new,channels))
    tmp_imgs = np.empty((test_Nimgs,height,width,channels))
    for path, subdirs, files in os.walk(imgs_dir): #list all files, directories in the path
        for i in range(len(files)):
            #original
            print ("original image: " +files[i])
            img = Image.open(imgs_dir+files[i])
            img=np.asarray(img)
            #img= img[500:3500,400:3400]
            img= img[0:1000,0:1000]
            tmp_imgs[i]=img
            
    tmp_imgs = np.transpose(tmp_imgs,(0,3,1,2))
    tmp_imgs = my_PreProc(tmp_imgs)

    imgs=get_testing_division(tmp_imgs, N_div)
    print ("imgs max: " +str(np.max(imgs)))
    print ("imgs min: " +str(np.min(imgs)))
    #reshaping for my standard tensors
    assert(imgs.shape == (test_Nimgs*N_div**2,channels,height_new,width_new))
    return imgs

def get_training_division(full_img,full_mask, N_div):
    full_img=np.asarray(full_img)
    full_mask=np.asarray(full_mask)
    div_h=int(full_img.shape[2]/(N_div))
    div_w=int(full_img.shape[3]/(N_div))
    divisions = np.empty((full_img.shape[0]*N_div**2,channels,div_h,div_w))
    divisions_masks=np.empty((full_img.shape[0]*N_div**2,1,div_h,div_w))
    
    for k in range(full_img.shape[0]):
        for i in range(N_div):
            for j in range(N_div):
                divisions[i*N_div+j+k*(N_div**2)] = full_img[k,:,i*div_h:(i+1)*div_h,j*div_w:(j+1)*div_w]
                divisions_masks[i*N_div+j+k*(N_div**2)]= full_mask[k,:,i*div_h:(i+1)*div_h,j*div_w:(j+1)*div_w]
    return divisions, divisions_masks

def get_testing_division(full_img, N_div):
    full_img=np.asarray(full_img)
    div_h=int(full_img.shape[2]/(N_div))
    div_w=int(full_img.shape[3]/(N_div))
    divisions = np.empty((full_img.shape[0]*N_div**2,channels,div_h,div_w))
    for k in range(full_img.shape[0]):
        for i in range(N_div):
            for j in range(N_div):
                divisions[i*N_div+j+k*(N_div**2)] = full_img[k,:,i*div_h:(i+1)*div_h,j*div_w:(j+1)*div_w]
    return divisions

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

imgs_train, groundTruth_train = get_datasets(original_imgs_train,groundTruth_imgs_train,"train")
print ("saving train datasets")
write_hdf5(imgs_train, dataset_path + "DRIVE_dataset_imgs_train.hdf5")
write_hdf5(groundTruth_train, dataset_path + "DRIVE_dataset_groundTruth_train.hdf5")


imgs_test= get_test_datasets(original_imgs_test,"test")
print ("saving test datasets")
write_hdf5(imgs_test,dataset_path + "DRIVE_dataset_imgs_test.hdf5")

#visualize(group_images(groundTruth_train,6),path_experiment+"test")#.show()
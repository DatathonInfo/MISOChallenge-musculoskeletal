import glob
import os
import shutil
import pydicom
import cv2
import json
import pandas as pd
import SimpleITK as sitk
import numpy as np
import pydicom as dicom
import sys
import argparse
from PIL import Image
from shapely.geometry import Polygon
from shapely.geometry import LinearRing
from skimage import draw
from matplotlib import pyplot as plt
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate, Activation,Conv2DTranspose
from keras.layers import BatchNormalization
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, History, EarlyStopping, ReduceLROnPlateau
from keras import backend as keras
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import r2_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def zero_padding(img):
    y, x = img.shape
    if x > y:
        new_size = x
        add_size = x-y
        
        add_image = np.zeros((new_size, add_size))
        if add_size % 2==0:
            add_img = np.zeros((int(add_size/2), new_size))
            img = np.concatenate((add_img,img, add_img),axis=0)

        elif add_size % 2==1:
            add_img_top = np.zeros((int(add_size/2),new_size))
            add_img_bot = np.zeros((int(add_size/2)+1, new_size))
            img = np.concatenate((add_img_top,img, add_img_bot),axis=0)


    elif y > x:
        new_size = y
        add_size = y-x
        
        add_image = np.zeros((add_size, new_size))
        if add_size % 2==0:
            add_img = np.zeros((new_size, int(add_size/2)))
            img = np.concatenate((add_img,img, add_img),axis=1)

        elif add_size % 2==1:
            add_img_left = np.zeros((new_size, int(add_size/2)))
            add_img_right = np.zeros((new_size, int(add_size/2)+1))
            img = np.concatenate((add_img_left,img, add_img_right),axis=1)
    
    return img

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def mkfolder(folder):
    if not os.path.lexists(folder):
        os.makedirs(folder)

def create_train_data(train_path, out_rows, out_cols, name, img_type):
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    
    imgs = glob.glob(train_path +"*." + img_type)
    imgdatas = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
    imglabels = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
    imgnames=[]
    for i, imgname in enumerate(imgs):
        if i%1000==0:
            print('{}/{}'.format(i, len(imgs)))
        midname = imgname[imgname.rindex("/")+1:-4]     
        img = load_img(imgname, color_mode = "grayscale")
        label = load_img(imgname.replace('jpg', 'png'), color_mode = "grayscale")
        img=img.resize((out_rows,out_cols))
        label=label.resize((out_rows,out_cols))

        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[i] = img
        imglabels[i] = label
        imgnames.append(midname)
    
    imgdatas = imgdatas.astype('uint8')
    imglabels = imglabels.astype('uint8')
    
    print('img : ', imgdatas.max())
    print('mask : ',imglabels.max())
    
    print('-'*30)
    print('normalization start...')
    print('-'*30)
    imgdatas = imgdatas/255.0
    
    imglabels[imglabels <= 127] = 0
    imglabels[imglabels > 127] = 1
    
    print('img : ',imgdatas.max())
    print('mask : ',imglabels.max())
    print('mask : ',imglabels.min())
    
    print('loading done')
    return(imgdatas,imglabels,imgnames)
    
def create_test_data(test_path, out_rows, out_cols, name, img_type):

    print('-'*30)
    print('Creating test images...')
    print('-'*30)

    i = 0
    imgs = glob.glob(test_path + "*." + img_type)
    imgdatas = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
    imglabels = np.ndarray((len(imgs),out_rows,out_cols,1), dtype=np.uint8)
    
    imgnames=[]
    for j, imgname in enumerate(imgs):
        if j%100==0:
            print('{}/{}'.format(j, len(imgs)))
        midname = imgname[imgname.rindex("/")+1:-4]
        img = load_img(imgname, color_mode = "grayscale")
        label = load_img(imgname.replace('jpg', 'png'), color_mode = "grayscale")
        img=img.resize((out_rows,out_cols))
        label=label.resize((out_rows,out_cols))

        img = img_to_array(img)
        label = img_to_array(label)
        imgdatas[j] = img
        imglabels[j] = label
        imgnames.append(midname)
         
    imgdatas = imgdatas.astype('uint8')
    imglabels = imglabels.astype('uint8')
    
    print('img : ', imgdatas.max())
    print('mask : ',imglabels.max())
    
    print('-'*30)
    print('normalization start...')
    print('-'*30)
    imgdatas = imgdatas/255.0
    
    imglabels[imglabels <= 127] = 0
    imglabels[imglabels > 127] = 1
    
    print('img : ',imgdatas.max())
    print('mask : ',imglabels.max())
    print('mask : ',imglabels.min())
    print('loading done')
    return(imgdatas,imglabels,imgnames)

def weight_center(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    cnt = contours[0] 
    mmt = cv2.moments(cnt) 
    cx = int(mmt['m10']/mmt['m00']) 
    cy = int(mmt['m01']/mmt['m00']) 
    print( 'x 무게중심', cx, 'y 무게중심', cy )
    return cx, cy
def randomRoate(img, label, p, angle_range):
    if len(img.shape)==3:
        height, width, channel = img.shape
    else:
        height, width = img.shape
    
    random_angle = np.random.randint(-angle_range/2, angle_range/2)*2
    matrix = cv2.getRotationMatrix2D((width/2, height/2), random_angle, 1)
    rotate_img = cv2.warpAffine(img, matrix, (width, height))
    rotate_label = cv2.warpAffine(label, matrix, (width, height))
    
    return rotate_img, rotate_label

def find_top_point(ori_img, mask_img):
    mask_img_T = mask_img.T

    tmp_index=[]
    for i, c in enumerate(mask_img_T):
        if len(np.unique(c))>1:
            tmp_index.append(i)
    x_min = min(tmp_index)
    x_max = max(tmp_index)

    p_x = int((x_min+x_max)/2) 
    p_y = np.where(mask_img_T[int((x_min+x_max)/2)]==255)[0][0]
    p = (p_x, p_y)

    return p_x, p_y

def Augment_crop(img, mask):
    p_x, p_y=find_top_point(img, mask)
    rotate_img, rotate_mask = randomRoate(img, mask, (p_x, p_y), 20)
    random_size = np.random.randint(15,35)*20 # 300-600
    
    h, w= img.shape
    x1 = p_x-random_size if p_x-random_size>0 else 0
    x2 = p_x+random_size if p_x+random_size<w else w
    y1 = p_y-random_size if p_y-random_size>0 else 0
    y2 = p_y+random_size if p_y+random_size<h else h
    
    crop_img = rotate_img[y1:y2,x1:x2]
    crop_mask = rotate_mask[y1:y2,x1:x2]

    return crop_img, crop_mask

def data_pre(dcmlist):
    for i in range(len(dcmlist)):
        # try:
        dicompath = dcmlist[i]
        dicom = pydicom.read_file(dicompath)
        img = dicom.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # except:
        #     print(dicompath)
        #     continue
        pad_img = zero_padding(img)
        image = Image.fromarray(pad_img)
        image = image.convert('L')
        image.save(dicompath.replace('.dcm','.jpg'))
    
    for dcm in dcmlist:
        jsonfile = dcm[:-4]+'.json'
        reader = sitk.ReadImage(dcm)
        image_array = sitk.GetArrayFromImage(reader)
        height = reader.GetMetaData('0028|0010')
        width = reader.GetMetaData('0028|0011')
        data = []
        for line in open(jsonfile,'r'):
            data.append(json.loads(line))
        for json_data in data:
            mask = np.zeros((int(height), int(width)))
            if json_data['annotation']['ANNOTATION_DATA'] is not None:
                for m in json_data['annotation']['ANNOTATION_DATA']:
                    if 'm_points' in m:
                        a = []
                        for i in m['m_points']:
                            b = (i['x'], i['y'])
                            a.append(b)
                        r = LinearRing(a)
                        s = Polygon(r)
                        x, y = s.exterior.coords.xy
                        maskd = poly2mask(y, x, (int(height), int(width)))
                        mask = mask + maskd
                mask = mask*255
                mask = zero_padding(mask)
                mask = np.expand_dims(mask, axis=0)
                img = sitk.GetImageFromArray(mask.astype('uint8'))
                num = 0
                maskpath = dcm[:-4]+'.png'
                sitk.WriteImage(img, maskpath)
            else:
                print('haha')


# In[13]:


def augmentation(img_path,mask_path):
    img_li = sorted(glob.glob(img_path))
    mask_li = sorted(glob.glob(mask_path))
    print(len(img_li), len(mask_li))
    i=0

    for img, mask in zip(img_li, mask_li):
        if i%100==0:
            print('{}/{}'.format(i, len(img_li)))
        savepath = 'train/aug/'
        mkfolder(savepath)
        ori_img = cv2.imread(img, 0)
        mask_img = cv2.imread(mask, 0)
        img_name = img[img.rindex('/')+1:-4]
        mask_name = img[mask.rindex('/')+1:-4]
        print(img_name)
        print(mask_name)
        cv2.imwrite(savepath+'/{}.jpg'.format(img_name), cv2.resize(ori_img, (512,512)))
        cv2.imwrite(savepath+'/{}.png'.format(img_name), cv2.resize(mask_img, (512,512)))
        for j in range(9):
            aug_img, aug_mask = Augment_crop(ori_img, mask_img)        
            aug_img = cv2.resize(aug_img, (512,512))
            aug_mask = cv2.resize(aug_mask, (512,512))
            cv2.imwrite(savepath+'/{}_{}.jpg'.format(img_name, j), aug_img)
            cv2.imwrite(savepath+'/{}_{}.png'.format(img_name, j), aug_mask)
        i+=1
    return savepath

def train_data_loading(path,image_size = 512):
    mkfolder(path)
    trainlist = glob.glob(path+'/*.dcm')
    data_pre(trainlist)
    train_img_path = path+'/*.jpg'
    train_mask_path = path+'/*.png'
    aug_path = augmentation(train_img_path,train_mask_path)
    imgs_train, imgs_mask_train, imgs_name = create_train_data(aug_path, image_size, image_size, 'train', 'jpg')
    return imgs_train, imgs_mask_train, imgs_name


def get_unet(img_rows, img_cols):
    inputs = Input((img_rows, img_cols,1))
    conv1 = Conv2D(32, (3, 3), activation=None, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, (3, 3), activation=None, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=None, padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, (3, 3), activation=None, padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=None, padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(128, (3, 3), activation=None, padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation=None, padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(256, (3, 3), activation=None, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation=None, padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(512, (3, 3), activation=None, padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation=None, padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(256, (3, 3), activation=None, padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation=None, padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(128, (3, 3), activation=None, padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation=None, padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, (3, 3), activation=None, padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation=None, padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(32, (3, 3), activation=None, padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

def dice_coef(y_true, y_pred, smooth=1e-6):
    print(y_pred)
    print(y_true)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def sens(y_true, y_pred): # sensitivity, recall
    print(y_pred)
    print(y_true)
    y_target_yn = K.round(K.clip(y_true, 0, 1)) # 실제값을 0(Negative) 또는 1(Positive)로 설정한다
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값을 0(Negative) 또는 1(Positive)로 설정한다

    # True Positive는 실제 값과 예측 값이 모두 1(Positive)인 경우이다
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 

    # (True Positive + False Negative) = 실제 값이 1(Positive) 전체
    count_true_positive_false_negative = K.sum(y_target_yn)

    # Recall =  (True Positive) / (True Positive + False Negative)
    # K.epsilon()는 'divide by zero error' 예방차원에서 작은 수를 더한다
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())

    # return a single tensor value
    return recall
def sch(epoch):
    if epoch>100 and epoch<=250:
        return 0.0001
    elif epoch>250:
        return 0.00001
    else:
        return 0.001


def deep(imgs_train,imgs_mask_train,path,batch_size = 4,epochs = 10,image_size=512):
    model = get_unet(image_size, image_size)
    model.summary()
    model = multi_gpu_model(model,gpus=2)
    model.compile(optimizer=Adam(lr=0.0001), loss=dice_coef_loss, 
                    metrics=['accuracy', sens, dice_coef_loss])
    
    check_model_path = path+'check/'
    predict_path = path+'pred/'
    mkfolder(check_model_path)
    mkfolder(predict_path)

    model_checkpoint = ModelCheckpoint(check_model_path+'ap_aug_exp1_{epoch:d}_{loss:f}.hdf5', 
                                        monitor='val_dice_coef_loss',verbose=1, 
                                        save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef_loss', factor=0.8, min_delta = 0.01, 
                                    patience=5, min_lr=1e-6, verbose=1)
    earlystopping = EarlyStopping(monitor='val_dice_coef_loss', patience=10)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    print('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=epochs, verbose=1, 
              validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, earlystopping])

    print('save model')
    model.save(predict_path+'ap_aug_exp1.h5')
    return model


def predict_save(pred_list,name_list):
    pred_img_path = 'train/pred/'
    if not os.path.isdir(pred_img_path):
        os.makedirs(pred_img_path)

    imgs = pred_list
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img[img <= 0.5] = 0
        img[img > 0.5] = 255
        img = array_to_img(img)
        img.save(pred_img_path+"%s_pred.png" %(name_list[i]))

def predict_val(model,test_path,image_size=512):
    test_list = glob.glob(test_path+'*.dcm')
    data_pre(test_list)
    imgs_test, imgs_label_test, test_name = create_test_data(test_path, image_size, image_size, 'test', 'jpg')
    print('predict test data')
    
    imgs_label_pred = model.predict(imgs_test, batch_size=4, verbose=1)
    name_list=test_name
    df = pd.DataFrame(columns=['name', 'acc', 'sen', 'spe', 'dsc'])

    true_list=imgs_label_test
    print(true_list.shape)

    pred_list=imgs_label_pred
    print(np.unique(pred_list))
    pred_list[pred_list > 0.5] = 1
    pred_list[pred_list <= 0.5] = 0
    
    sensitivity=[]
    specificity=[]
    acc=[]
    dsc=[]

    for i in range(len(true_list)):
        yt=true_list[i].flatten()
        yp=pred_list[i].flatten()
        mat=confusion_matrix(yt,yp)
        if len(mat) == 2:
            ac=(mat[1,1]+mat[0,0])/(mat[1,0]+mat[1,1]+mat[0,1]+mat[0,0])
            st=mat[1,1]/(mat[1,0]+mat[1,1])
            sp=mat[0,0]/(mat[0,1]+mat[0,0])
            if mat[1,0]+mat[1,1] == 0:
                specificity.append(sp)
                acc.append(ac)
            else:
                sensitivity.append(st)  
                specificity.append(sp)
                acc.append(ac)
        else:
            specificity.append(1)
            acc.append(1)

        yt=true_list[i]
        yp=pred_list[i]
        if np.sum(yt) != 0 and np.sum(yp) != 0:
            dice = np.sum(yp[yt==1])*2.0 / (np.sum(yt) + np.sum(yp))
            dsc.append(dice)
        df=  df.append({'name':name_list[i], 'acc':ac, 'sen':st, 'spe':sp, 'dsc':dice}, ignore_index=True)

    print("complete")      
    print("acc avg : {0:0.4f}".format(np.mean(acc)))
    print("sensitivity avg : {0:0.4f}".format(np.mean(sensitivity)))
    print("specificity avg : {0:0.4f}".format(np.mean(specificity)))
    print("dsc avg : {0:0.4f}".format(np.mean(dsc)))

    predict_save(pred_list,name_list)
    return test_name

def get_results(test_name):
    #test_name 순서대로
    #4,5번 사이의 각도 측정해서 list로 저장
    #그리고 4,5번 척추 사이의 거리 측정해서 list로 따로 저장
    #부탁드립니다
    angle_list = list(np.empty(shape=(60,), dtype=np.int8))
    dist_list = list(np.empty(shape=(60,), dtype=np.int8))
    return angle_list,dist_list #predict 결과들의 Angle 측정값


def get_score(angle_list,dist_list,test_name,test_path):
    angle_test = []
    angle_ai = []
    dist_test = []
    dist_ai = []
    get_no45 = []
    for i in range(len(test_name)):
        name = test_name[i]
        data = []
        
        jsonfile = test_path+'/{}.json'.format(name)
        for line in open(jsonfile,'r'):
            data.append(json.loads(line))
        for json_data in data:
            check = 0
            if json_data['annotation']['ANNOTATION_DATA'] is not None:
                for m in json_data['annotation']['ANNOTATION_DATA']:
                    if m['type']=='cobbAngle':
                        if m['label'] == 'L4-5A':
                            angle_test.append(m['angle'])
                            angle_ai.append(angle_list[i])
                            check = 1
                    elif m['type']=='line':
                        if m['label'] == 'L4-5H':
                            dist_test.append(m['distMm'])
                            dist_ai.append(dist_list[i])
                            check = 1
                if check==0:
                    get_no45.append(name)
    print(get_no45)
    print(r2_score(angle_ai, angle_test))
    print(r2_score(dist_ai, dist_test))


def main():
    parser = argparse.ArgumentParser(description='Parser test input uid')
    parser.add_argument('--train_path', type = str, default ='train/', help='학습데이터 위치')
    parser.add_argument('--test_path', type = str, default ='test/', help='테스트데이터 위치')
    parser.add_argument("--image_size",type= int , default= 512, help='학습에 사용될 이미지의 크기')
    parser.add_argument("--epochs",type= int , default= 10, help='에폭')
    parser.add_argument("--batch_size",type= int , default= 4, help='배치사이즈')
    args = parser.parse_args()
    
    imgs_train, imgs_mask_train, imgs_name = train_data_loading(args.train_path, image_size = args.image_size)
    model = deep(imgs_train,imgs_mask_train,args.train_path,batch_size = args.batch_size,epochs = args.epochs,image_size=args.image_size)
    test_name = predict_val(model,args.test_path,image_size = args.image_size)
    angle_list,dist_list = get_results(test_name)
    get_score(angle_list,dist_list,test_name,args.test_path)
    
if __name__ == "__main__":
    main()


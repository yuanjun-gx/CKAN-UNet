import os
import random
import numpy as np
from PIL import Image

root_dir = "" # change it in your saved original data path
save_dir = ""


if __name__ == '__main__':
    imgfile = os.path.join(root_dir, 'images')
    labfile = os.path.join(root_dir, 'masks')
    filename = sorted([os.path.join(imgfile, x) for x in os.listdir(imgfile) if x.endswith('.png')])
    random.shuffle(filename)
    labname = [filename[x].replace('ISIC2018_Task1-2_Training_Input', 'ISIC2018_Task1_Training_GroundTruth'
                                   ).replace('.jpg', '_segmentation.png') for x in range(len(filename))]

    # labname = [os.path.join(labfile, x) for x in os.listdir(labfile) if x.endswith('.png')]


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir+'/image')
        os.makedirs(save_dir+'/label')
    
    kk = 0
    for i in range(len(filename)):
        fname = filename[i].rsplit('/', maxsplit=1)[-1].split('.')[0]
        lname = labname[i].rsplit('/', maxsplit=1)[-1].split('.')[0]

        image = Image.open(filename[i])
        label = Image.open(labname[i])
        label = label.convert('L')


        image = image.resize((224, 224))
        label = label.resize((224, 224))

        image = np.array(image)
        label = np.array(label)

        images_img_filename = os.path.join(save_dir, 'image', fname)
        labels_img_filename = os.path.join(save_dir, 'label', lname+'_segmentation')
        np.save(images_img_filename, image)
        np.save(labels_img_filename, label)
        kk += 1
        print("KK: ", kk)
        
    print('Successfully saved preprocessed data')

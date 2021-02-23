# Originaldatei s. https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task120_Massachusetts_RoadSegm.py
# angepasst und veraendert auf unsere Datensaetze

import numpy as np
import random
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti


if __name__ == '__main__':

    # download dataset from https://www.kaggle.com/insaff/massachusetts-roads-dataset
    # extract the zip file, then set the following path according to your system:
    base = '/home/uni/AutoML-Share/NNUnet/2d_to_nifti'
    
    
    
    
    # Dort muessen die Ordner 2d/imgs und 2d/labels vorhanden sein

    # now start the conversion to nnU-Net:
    task_name = 'Task205_Augen_weniger-Test'
    
    
    target_base = join(nnUNet_raw_data, task_name)  # nnUNet Umgebungsvariablen muessen gesetzt sein
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTs)

    # convert the training examples.
    labels_dir_tr = join(base, '2d', 'labels')
    images_dir_tr = join(base, '2d', 'imgs')
    training_cases = subfiles(labels_dir_tr, suffix='.png', join=False)
    for t in training_cases:
        unique_name = t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_segmentation_file = join(labels_dir_tr, t)
        input_image_file = join(images_dir_tr, t)
        # Splitting des Datasets in Train und Test
        if random.randint(0,3) >= 1: # 2 drittel Train, 1 drittel Test
            # TRAIN-SPLIT
            output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
            output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you
        else:
            # TEST-SPLIT
            output_image_file = join(target_imagesTs, unique_name)
            output_seg_file = join(target_labelsTs, unique_name)
        
        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x!=0).astype(int))
	
    # finally we can call the utility for generating a dataset.json
    # Retina 2D
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Rot','Gruen','Blau'), labels={1: 'Ader'}, dataset_name=task_name, license='Lizenz')
    
    # Pascal VOC 2012
    #generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Rot','Gruen','Blau'), labels={1: 'Aeroplane', 2: 'Bicycle', 3: 'Bird', 4: 'Boat', 5: 'Bottle', 6: 'Bus', 7: 'Car', 8: 'Cat', 9: 'Chair', 10: 'Cow', 11: 'Diningtable', 12: 'Dog', 13: 'Horse', 14: 'Motorbike', 15: 'Person', 16: 'Pottedplant', 17: 'Sheep', 18: 'Sofa', 19: 'Train', 20: 'TVmonitor'}, dataset_name=task_name, license='Lizenz')

    """
    once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the 
    `nnUNet_plan_and_preprocess` command like this:
    
    > nnUNet_plan_and_preprocess -t 120 -pl3d None
    
    once that is completed, you can run the trainings as follows:
    > nnUNet_train 2d nnUNetTrainerV2 120 FOLD
    
    (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)
    
    there is no need to run nnUNet_find_best_configuration because there is only one model to shoose from.
    Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    `nnUNet_determine_postprocessing` command
    """
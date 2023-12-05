import os
from pathlib import Path
import glob

def create_dir(dir_name_main:str):
    if os.path.isdir(dir_name_main):
        print('Directory already exists')
    else:
        path=Path('Brain-Tumor-Detection/datasets/'+dir_name_main)
        os.makedirs(path,exist_ok=True)
        path_no=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/no')
        os.makedirs(path_no,exist_ok=True)
        path_yes=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/yes')
        os.makedirs(path_yes,exist_ok=True)
        path_test=Path('Brain-Tumor-Detection/datasets/'+dir_name_main+'/test')
        os.makedirs(path_test,exist_ok=True)
        print('Creating directory')


def create_dataset(paths:[],dir_name_main:str):
    create_dir(dir_name_main)
    for path in paths:
        print(path)
        for subdir, dirs, files in os.walk(path):
            for dir_name in dirs:
                if dir_name=='no':
                        print('if',dir_name)
                        for subdir, dirs, files in os.walk(path+'/'+dir_name):
                            print('Files',subdir, dirs, files)
                            for file in files:
                                print('Image',file)
                                os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/no/{file}")
                elif dir=='yes':
                        for subdir, dirs, files in os.walk(path+'/'+dir_name):
                                for file in files:
                                    os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/yes/{file}")
                elif dir=='pred':
                        for subdir, dirs, files in os.walk(path+'/'+dir_name):
                                for file in files:
                                    os.replace(f"{path}/{dir_name}/{file}",f"Brain-Tumor-Detection/datasets/{dir_name_main}/test/{file}")


create_dataset(paths=['Brain-Tumor-Detection/datasets/brain_tumor_dataset','Brain-Tumor-Detection/datasets/Brain_Tumor_Detection'],dir_name_main='segmentation_dataset')
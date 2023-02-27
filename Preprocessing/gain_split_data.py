import os
import SimpleITK
import torchio as tio
import nibabel as nib


data_root = r'E:\NPCICdataset\Patient_Image\test_intro_vae_result'
ori_data_name = '001_ori_register_nii'
patch_size = 64

sub_id = []
sub_path = []
for file_name in os.listdir(os.path.join(data_root, ori_data_name)):
    sub_id.append(file_name)
    sub_path.append(os.path.join(os.path.join(data_root, ori_data_name, file_name)))
##根据对应的图像，先去生成对应的t1c的图像来看下效果






##use t1 for the spilt the data of the result


## 根据对应的图像将对应的进行保存以及分层


## 进行对应的图像文件的生成以及对应的保存
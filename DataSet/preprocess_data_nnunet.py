from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import os

join = os.path.join
isdir = os.path.isdir
isfile = os.path.isfile
listdir = os.listdir
makedirs = maybe_mkdir_p
os_split_path = os.path.split


def prepare_data(ori_path,data_path,label_path):
    ##将原始的数据整理成下面代码需要的形式
    os.walk()

if __name__ == "__main__":
    """
    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """
    task_name = "Task001_npc"
    data_dir = r"/home/amax/Task/nnUnet_data/ori_data/data"                                      # 原始数据存放路径
    label_dir = r"/home/amax/Task/nnUnet_data/ori_data/labels"                                   # 原始标签存路径
    nnUNet_raw_data = r'/home/amax/Task/nnUnet_data/nnUNet_raw_data_base/nnUNet_raw_data'        # 整理后数据存放路径

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)

    patient_names = []
    for patient_name in subdirs(data_dir, join=False):
        patdir = join(data_dir, patient_name)
        patdir_lab = join(label_dir, patient_name)
        patient_names.append(patient_name)

        # 不同模态数据的原始路径
        ct = join(patdir, "ct_Mask.nii.gz")
        t1 = join(patdir, "t1.nii.gz")
        t1c = join(patdir, "t1c.nii.gz")
        t2 = join(patdir, "t2.nii.gz")
        t1dix = join(patdir, "T1DIXONC.nii.gz")
        seg = join(patdir_lab, "label.nii.gz")

        assert all([
            isfile(ct),
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(t1dix),
            isfile(seg)
        ]), "%s" % patient_name

        # 将数据按照标准格式命名并复制到指定路径中
        shutil.copy(ct, join(target_imagesTr, "NPC_" + patient_name + "_0000.nii.gz"))
        shutil.copy(t1, join(target_imagesTr, "NPC_" + patient_name + "_0001.nii.gz"))
        shutil.copy(t1c, join(target_imagesTr, "NPC_" + patient_name + "_0002.nii.gz"))
        shutil.copy(t2, join(target_imagesTr, "NPC_" + patient_name + "_0003.nii.gz"))
        shutil.copy(t1dix, join(target_imagesTr, "NPC_" + patient_name + "_0004.nii.gz"))
        shutil.copy(seg, join(target_labelsTr, "NPC_" + patient_name + ".nii.gz"))

    # 创建json文件内容
    json_dict = OrderedDict()
    json_dict['name'] = "Npc_Seg"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "nothing"
    json_dict['licence'] = "nothing"
    json_dict['release'] = "0.0"

    # 不同模态对应的ID
    json_dict['modality'] = {
        "0": "CT",
        "1": "T1",
        "2": "T1C",
        "3": "T2",
        "4": "T1DIXONC"
    }
    # 不同标签对应的one-hot码
    json_dict['labels'] = {
        "0": "background",
        "1": "GTVnx",
        "2": "Spinal_cord",
        "3": "Brain_Stem",
        "4": "Parotid"
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/NPC_%s.nii.gz" % i, "label": "./labelsTr/NPC_%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = []

    # 将字典写入json文件中
    save_json(json_dict, join(target_base, "dataset.json"))

import SimpleITK as sitk
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
from collections import OrderedDict

def get_patient_identifiers_from_cropped_files(folder):
    return [i.split("/")[-1][:-4] for i in subfiles(folder, join=True, suffix=".npz")]

def get_case_identifier_from_npz_win(case):
    case_identifier = case.split("\\")[-1]
    return case_identifier

def get_case_from_whole_name(case_inden):
    return [get_case_identifier_from_npz_win(i) for i in case_inden]

def main():


    folder = r'E:\NPCICdataset\Patient_Image\seg_test\nnUNet_process\Task001_Npc'
    train_pkl_ad = r'E:\NPCICdataset\Patient_Image\seg_test\nnUNet_raw_data\Task001_Npc\spilt' + '.pkl'
    case_indentifier = get_patient_identifiers_from_cropped_files(folder)
    list_name = get_case_from_whole_name(case_indentifier)
    write_pickle(list_name, train_pkl_ad)
    print(list_name)



if __name__=='__main__':
    main()

import torch
import torch.nn.functional as F
import platform
import scipy.io as scio



# 1.2 user setting for the platfrom
user = "hanxv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bs_linux = 6
bs_windows = 2

if platform.system() == "Linux" and user == "yanghening":
    train_root_dir = '/public2/yanghening/hanxv_exp/DataSet_nii/train_gp_mask_new'
    val_root_dir = '/public2/yanghening/hanxv_exp/DataSet_nii/val_gp_mask_new'
    label_path = '/public2/yanghening/hanxv_exp/DataSet_nii/label.mat'

    train_labels = scio.loadmat(label_path)["label_train"]
    val_labels = scio.loadmat(label_path)["label_test"]
    bs = bs_linux

elif platform.system() == "Linux" and user == "hanxv":

    train_root_dir = '/public2/hanxv/EBVexp/DataSet/ZhongShan_nii/ZhongShan_gp_nii_register/train_gp_mask_new'
    val_root_dir = '/public2/hanxv/EBVexp/DataSet/ZhongShan_nii/ZhongShan_gp_nii_register/val_gp_mask_new'
    label_path = '/public2/hanxv/EBVexp/DataSet/ZhongShan/label.mat'

    train_labels = scio.loadmat(label_path)["label_train"]
    val_labels = scio.loadmat(label_path)["label_test"]
    bs = bs_linux

elif platform.system() == "Windows":
    train_labels = scio.loadmat(r"F:\EBV_dataset\ZhongShan_gp_nii_register\label.mat")["label_train"]
    val_labels = scio.loadmat(r"F:\EBV_dataset\ZhongShan_gp_nii_register\label.mat")["label_test"]
    train_root_dir = r'F:\EBV_dataset\ZhongShan_gp_nii_register\train_gp_mask_new'
    val_root_dir = r'F:\EBV_dataset\ZhongShan_gp_nii_register\val_gp_mask_new'
    bs = bs_windows
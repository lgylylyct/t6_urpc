import platform


class Args:
    def __init__(self) -> None:
        pass


args = Args()

# supervised
args.exp = "Sup_Swin_T1C"
args.semi_sup = False

# model
args.model = "unet_urpc"

# training
args.max_iterations = 9000
args.base_lr = 0.01
args.optim = 'SGD'


# dataset
args.mod = "T1C"
args.label_mod_type = "T1C_mask"
args.batch_size = 8
args.labeled_bs = 4
args.labeled_num = 60
args.num_classes = 2
args.patch_size = [224, 224]
if platform.system() == "Windows":
    args.img_path = "I:/linguoyu/DataSet/NPCICDataset/zhongshan1/005_ToNumpy"
    args.mask_path = "I:/linguoyu/DataSet/NPCICDataset/zhongshan1/803_GT_R_Resample_numpy"
elif platform.system() == "Linux":
    args.img_path = "/public/linguoyu/PythonProgramme/DataSet/NPCIC/zhongshan1/005_ToNumpy"
    args.mask_path = (
        "/public/linguoyu/PythonProgramme/DataSet/NPCIC/zhongshan1/803_GT_R_Resample_numpy"
    )

# consistency
args.consistency = 1.0
args.consistency_rampup = 100.0


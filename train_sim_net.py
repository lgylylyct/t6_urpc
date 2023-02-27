from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord, Resized, Spacingd, \
    ScaleIntensityRanged
import glob
import os
import torch
import datetime
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
import monai
import scipy.io as scio
import numpy as np
import torch.nn.functional as F
import platform
from monai.data import Dataset
from initial_log import get_logger
import torch.backends.cudnn as cudnn
from torchsummary import summary
from Model.create_model import create_model_backbone
os.environ['CUDA_VISIBLE_DEVICES'] = "4"




def random_all(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

def log_info(log_path, par_name):
    logger = get_logger(log_path)
    logger.info(par_name)

    return logger

def main():
    #1.1   parameter setting
    test_or_not = False
    freeze = True
    par_name = "renset50_tencent_medical_change_something_wrong(acc_auc) and the pixel size is 222"  ##you want to tell youself in this exp
    log_path = os.path.join(os.getcwd(), ('log_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'))

    epochs = 500
    auc_threshold = 0.6
    learning_rate = 1e-4
    user = "hanxv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    ## whther use the model weight of the resnet
    use_pretrain = False
    model_weight_path = "./resnet50_23_new.pth"
    test_dataloader = False
    print_or_no = False

    random_seed = 11

    log_or_no = True
    bs_linux = 6
    bs_windows = 2


    model_name = "resnet"
    model_layer_num = 50
    n_input_channels = 1
    n_classes = 2

    if test_or_not:
        pix_dim = (10, 10, 10)
        spatial_size = (10, 10, 10)
        train_num = 24
        val_num = 40
        #(256,256,32)
    else:
        train_num = 420
        val_num = 106
        pix_dim = (2, 2, 2)
        spatial_size = (256, 256, 32)


    sub_t1c_train = []
    sub_t1c_val = []
    save_path = './use_pre_train_renset50_freeze.pth'
    best_acc = 0.0




    #1.2 user setting for the platfrom
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


    random_all(random_seed)


    ## 2.1 gain the subfile_path of the file
    logger = log_info(log_path, par_name)

    sub_train_dir = glob.glob(train_root_dir + "/*")
    sub_val_dir = glob.glob(val_root_dir + "/*")


    for sub_train_dir in sub_train_dir:
        sub_t1c_train_ = os.path.join(sub_train_dir, 'addmask.nii')
        sub_t1c_train.append(sub_t1c_train_)

    for sub_val_dir in sub_val_dir:
        sub_t1c_val_ = os.path.join(sub_val_dir, 'addmask.nii')
        sub_t1c_val.append(sub_t1c_val_)

    train_images = sub_t1c_train
    val_images = sub_t1c_val

    train_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    val_data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(val_images, val_labels)
    ]

    transform = Compose(
        [
            LoadImaged(keys='image'),                       # ScaleIntensity(),
            ScaleIntensityRanged(keys='image', a_min=-300.0, a_max=300.0),
            AddChanneld(keys='image'),
            #RepeatChanneld(keys='image', repeats=3),
            Spacingd(keys='image', pixdim=pix_dim),
            Resized(keys='image', spatial_size=spatial_size),
            ToTensord(keys='image')
        ]
    )

    train_ds = Dataset(data=train_data_dicts[:train_num], transform=transform)
    val_ds = Dataset(data=val_data_dicts[:val_num], transform=transform)

    train_loader = monai.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=0)
    test_loader = monai.data.DataLoader(
        val_ds, batch_size=bs, shuffle=True, num_workers=0)

    if test_dataloader:
        test_image = next(iter(train_loader))
        tt = test_image['image']
        t_shape = tt.shape

    train_mr_num, val_mr_num = len(train_ds), len(val_ds)


    print("using {} mr for training, {} mr for validation.".format(train_mr_num, val_mr_num))
    if log_or_no:
        logger.info("using {} mr for training, {} mr for validation.".format(train_mr_num, val_mr_num))

    # 3.1 define the model of the cal

    model = create_model_backbone(model_name=model_name, model_layer_num=model_layer_num, print_or_no=print_or_no,
                                  freeze=freeze,n_classes=n_classes,n_input_channels=n_input_channels)


    #3.2 use the train for the dataset
    if use_pretrain:
        pretext_weight = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(pretext_weight, strict=False)

    model.to(device)

    loss_function = torch.nn.BCEWithLogitsLoss()
    params = [p1 for p1 in model.parameters() if p1.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # ############################### train the whole model
    # ##################################################

    ###4.1 train the model ########################
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        acc_train = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        outputs_prob_train = []
        outputs_prob_train_2 = []
        gt_train_all = []

        for item in train_bar:
            images = item["image"]
            labels = item["label"].squeeze()
            optimizer.zero_grad()

            logits = model(images.to(device))


            labels = labels.to(device).squeeze()

            one_hot_labels = torch.eye(2)[labels.unsqueeze(dim=0).long(), :]
            one_hot_labels = one_hot_labels.to(device)

            loss = loss_function(logits, one_hot_labels.squeeze(dim=0))

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predict_y_train = torch.max(logits, dim=1)[1]
            prob_train = F.softmax(logits, dim=1)

            gt_train = labels.detach().cpu().numpy()
            gt_list = gt_train.tolist()
            gt_train_all.extend(gt_list)

            outputs_prob_train.extend(prob_train)
            outputs_prob_train_2.extend(prob_train.data[:, 1].detach().cpu().numpy())

            acc_train += torch.eq(predict_y_train, labels.to(device)).sum().item()/bs

            train_bar.desc = "train epoch[{}/{}] loss:{:.6f}".format(epoch + 1, epochs, loss.item())


        train_accurate = acc_train / len(train_ds)
        train_auc_ = roc_auc_score(gt_train_all, outputs_prob_train_2)

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  train_auc_: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, train_auc_))
        if log_or_no:
            logger.info('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  train_auc_: %.3f' %
                  (epoch + 1, running_loss / train_steps, train_accurate, train_auc_))

        ##val the model##
        model.eval()
        acc_val = 0.0
        outputs_prob = []
        gt_val_all = []  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader, file=sys.stdout)
            for val_item in val_bar:
                val_images = val_item["image"]
                val_labels = val_item["label"]
                outputs = model(val_images.to(device))

                predict_y = torch.max(outputs, dim=1)[1]
                prob = F.softmax(outputs, dim=1)

                acc_val += torch.eq(predict_y, val_labels.to(device)).sum().item()/bs
                gt_val = val_labels.detach().cpu().numpy()
                gt_val_list = gt_val.tolist()

                gt_val_all.extend(gt_val_list)
                outputs_prob.extend(prob[:, 1].cpu().numpy())

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc_val / len(val_ds)
        val_auc = roc_auc_score(gt_val_all, outputs_prob)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  val_auc: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate, val_auc))
        if log_or_no:
            logger.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  val_auc: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate, val_auc))



        ##5.1 the metric of the save
        if val_accurate > best_acc and val_auc > auc_threshold:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main()


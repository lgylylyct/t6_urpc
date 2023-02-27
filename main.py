from monai.transforms import Compose, LoadImaged, AddChanneld, ToTensord, Resized, Spacingd, \
    ScaleIntensityRanged
import glob
import os
import torch
import datetime
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
import monai
import scipy.io as scio
import torch.nn.functional as F
import platform
from monai.data import Dataset
from initial_log import get_logger
from Model.model3d import generate_model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    #1.1   parameter setting
    epochs = 500
    auc_threshold = 0.6
    bs = 1
    train_num = 42
    val_num = 40
    learning_rate = 1e-4
    user = "hanxv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight_path = "./resnet50_23_new.pth"
    log_path = os.path.join(os.getcwd(), ('log_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'))
    model_type = 1
    use_pretrain = True
    pix_dim = (10, 10, 10)
    spatial_size = (10, 10, 10)  #(256,256,32)
    test_dataloader = False
    train_type = 2  ## 2:transfer 1: scratch
    par_name = "renset50_tencent_medical"


    #1.2 user setting for the platfrom
    if platform.system() == "Linux" and user == "yanghening":
        train_root_dir = '/public2/yanghening/hanxv_exp/DataSet_nii/train_gp_mask_new'
        val_root_dir = '/public2/yanghening/hanxv_exp/DataSet_nii/val_gp_mask_new'

        label_path = '/public2/yanghening/hanxv_exp/DataSet_nii/label.mat'

        train_labels = scio.loadmat(label_path)["label_train"]
        val_labels = scio.loadmat(label_path)["label_test"]
        bs = 6

    elif platform.system() == "Linux" and user == "hanxv":

        train_root_dir = '/public2/hanxv/EBVexp/DataSet/ZhongShan_nii/ZhongShan_gp_nii_register/train_gp_mask_new'
        val_root_dir = '/public2/hanxv/EBVexp/DataSet/ZhongShan_nii/ZhongShan_gp_nii_register/val_gp_mask_new'

        label_path = '/public2/hanxv/EBVexp/DataSet/ZhongShan/label.mat'

        train_labels = scio.loadmat(label_path)["label_train"]
        val_labels = scio.loadmat(label_path)["label_test"]
        bs = 6

    elif platform.system() == "Windows":
        train_labels = scio.loadmat(r"F:\EBV_dataset\ZhongShan_gp_nii_register\label.mat")["label_train"]
        val_labels = scio.loadmat(r"F:\EBV_dataset\ZhongShan_gp_nii_register\label.mat")["label_test"]
        train_root_dir = r'F:\EBV_dataset\ZhongShan_gp_nii_register\train_gp_mask_new'
        val_root_dir = r'F:\EBV_dataset\ZhongShan_gp_nii_register\val_gp_mask_new'
        bs = 1


    ## 2.1 gain the subfile_path of the file

    logger = get_logger(log_path)
    logger.info(par_name)

    sub_train_dir = glob.glob(train_root_dir + "/*")
    sub_val_dir = glob.glob(val_root_dir + "/*")

    sub_t1c_train = []
    sub_t1c_val = []
    save_path = './desnet1.pth'
    best_acc = 0.0

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
            ToTensord(keys='image')  # 把图像转成tensor格式
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
    logger.info("using {} mr for training, {} mr for validation.".format(train_mr_num, val_mr_num))

    # 3.1 define the model of the cal
    if model_type == 3:
        model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=2, init_features=64,
                                             growth_rate=32,
                                             block_config=(6, 12, 24, 16), bn_size=4, act=('relu', {'inplace': True}),
                                             norm='batch', dropout_prob=0.0)
    if model_type == 2:

        model = monai.networks.nets.DenseNet264(init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), pretrained=False,
                                        progress=True,spatial_dims=3, in_channels=1, out_channels=2)

    if model_type == 1:
        # model = monai.networks.nets.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
        # model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)
        model = generate_model(50, n_input_channels=1, n_classes=2)


    #3.2 use the train for the dataset
    if use_pretrain:
        pretext_weight = torch.load(model_weight_path, map_location=device)
        model.load_state_dict(pretext_weight, strict=False)

    model.to(device)
    # model = monai.networks.nets.RE
    loss_function = torch.nn.BCEWithLogitsLoss()

    params = [p1 for p1 in model.parameters() if p1.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # ##############################################  train the whole model
    # ##################################################

    ###2.Similarity from the 1########################
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        acc_train = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        outputs_prob_train = []
        outputs_prob_train_1 = []
        outputs_prob_train_2 = []
        gt_train_all = []
        i = 0
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
            # gt_train_all.extend(gt_list)
            gt_train_all.append(gt_list)
            outputs_prob_train.append(prob_train)
            outputs_prob_train_2.append(prob_train.data[:, 1].detach().cpu().numpy())

            ######--------------计算ACC和AUC相关的�?            acc_train += torch.eq(predict_y_train, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.6f}".format(epoch + 1, epochs, loss.item())

        train_accurate = acc_train / len(train_ds)
        train_auc_ = roc_auc_score(gt_train_all, outputs_prob_train_2)

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  train_auc_: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, train_auc_))
        logger.info('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  train_auc_: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, train_auc_))

        ###########################################################      validate2      #################################################################################
        model.eval()
        acc = 0.0
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

                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                gt_val = val_labels.detach().cpu().numpy()
                gt_val_list = gt_val.tolist()

                gt_val_all.extend(gt_val_list)
                outputs_prob.extend(prob[:, 1].cpu().numpy())

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / len(val_ds) / 2
        # val_auc = roc_auc_score(val_mr_data_set.mr_label, outputs_prob)
        val_auc = roc_auc_score(gt_val_all, outputs_prob)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  val_auc: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate, val_auc))
        logger.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  val_auc: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate, val_auc))
        ###############################################################save metric#########################################################################
        ##3.1 the metric of the save
        if val_accurate > best_acc and val_auc > auc_threshold:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()


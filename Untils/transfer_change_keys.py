import os
import json
import sys
import math
import random
import torch
import platform
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import generate_model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


'''
处理对应的权重以及对应的结果值
'''

####################################################give the para of the model and the dataset #################
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net2 = generate_model(50, n_input_channels=1, n_classes=2)
    ###############################     load pretrain weights   ################################

    model_weight_path1 = "./resnet101-pre.pth"
    model_weight_path50 = "./r3d50_S_200ep.pth"
    model_weight_path_23 = './resnet_50_23dataset.pth'

    assert os.path.exists(model_weight_path50), 'file {} dose not exist.'.format(model_weight_path50)
    p_w = torch.load(model_weight_path50,map_location=device)

    p_w_u = p_w["state_dict"]


    p_w_ =torch.load(model_weight_path_23,map_location=device)
    p_w_u_n = p_w_["state_dict"]



    for key,value in p_w_u_n.items():
        p=value


    m_101_w = net2.state_dict()

    del_key = []
    for key, _ in p_w_u.items():
        if "fc" in key:
            del_key.append(key)

    for key in del_key:
        del p_w_u[key]

    dict_new = dict(zip(p_w_u.keys(), p_w_u_n.values()))
    ##因为这个名字的问题就交换key和value值之间的名字
    torch.save(dict_new, "./resnet50_23_new.pth")


    # missing_keys, unexpected_keys = net2.load_state_dict(p_w_u,strict=False)
    # print("[missing_keys]", *missing_keys, sep="\n")
    # print("[unexpected_keys]", *unexpected_keys, sep="\n")
    # torch.save(p_w_u, "./ch_resnet50_video_c3.pth")



    net2.load_state_dict(dict_new,strict=False)


    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure

    # in_channel50 = net2.fc.in_features
    #
    #
    # net2.fc = nn.Linear(in_channel50, 2)
    # net2.to(device)
    #
    # #############################################    define loss function      ###################################################
    # loss_function = nn.CrossEntropyLoss()
    # loss_function2 = nn.BCELoss()
    #
    # params2 = [p2 for p2 in net2.parameters() if p2.requires_grad]
    #
    #
    # optimizer2 = optim.AdamW(params2, lr=0.001)





if __name__ == '__main__':
    main()

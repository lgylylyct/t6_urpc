from typing import Union
#import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import os, cv2
import torch
import re, json
import pickle as pkl
import socket


def writeJson(data_dict, path):
    with open(path, "w") as f:
        f.write(json.dumps(data_dict))


def readJson(path):
    with open(path, "r") as f:
        data_dict = dict(json.load(f))
    return data_dict


def save_to_pkl(path, data):
    with open(path, "wb") as f:
        pkl.dump(data, f)


def load_from_pkl(path):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


def makeDirs(path):
    path_ = path.replace("../", "")
    path_ = path_.replace("./", "")

    if "." not in path_:
        path = path
    else:
        path = os.path.split(path)[0]

    if not os.path.exists(path):
        os.makedirs(path)


def DFAdd2CSV(df, data, columns, csv_path="log.csv", index=None):
    save_dir = os.path.split(csv_path)[0]
    if not os.path.exists(save_dir) and save_dir != "":
        os.makedirs(save_dir)

    if df is None:
        df = pd.DataFrame(index=index)
    assert isinstance(data, list), "data must be 1D list or 2D list"
    if not isinstance(data[0], list):
        data = [data]
    _df = pd.DataFrame(data=data, columns=columns, index=index)
    df = pd.concat([df, _df], ignore_index=index is None)
    df.to_csv(csv_path, index=index is not None, encoding="utf_8_sig")
    return df


def CSV2Png(csv_path="log.csv", png_path="log.png"):
    c = ["b", "g", "r", "c", "m", "y", "k", "w"]
    df = pd.read_csv(csv_path)
    labels = list(df.columns)
    assert len(c) >= len(labels), "the number of pd.columes must < 8"
    for i in range(1, len(labels)):
        plt.plot(df.iloc[:, 0], df.iloc[:, i], label=labels[i], c=c[i])
    plt.legend()
    plt.savefig(png_path)


def toTensor(ndarray, dtype=torch.float32, add_channel=0):
    if not isinstance(ndarray, np.ndarray):
        return ndarray

    for i in range(add_channel):
        ndarray = ndarray[np.newaxis, ...]
    tensor = torch.as_tensor(ndarray.copy(), dtype=dtype)
    return tensor


def toNumpy(tensor, dtype=np.float32, is_squeeze=True):
    if not isinstance(tensor, torch.Tensor):
        return tensor

    if is_squeeze:
        tensor = torch.squeeze(tensor)
    ndarray = tensor.detach().cpu().numpy().astype(dtype)
    return ndarray


def to_255_gray_level(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    arr = arr.astype(np.uint8)
    return arr


def conditionalIterInfo(i, steps_per_epoch, medium):
    if steps_per_epoch < 4:
        return True

    if (
        (i + 1) % (steps_per_epoch // 4) == 0
        or i + 1 == steps_per_epoch
        or i + 1 == medium["config"]["debug_iteration"]
    ):
        return True
    else:
        False


def printLogMetrics(log_metrics, medium):
    for k, names in medium["config"]["log_metrics"].items():
        values = log_metrics[k]

        if len(values) < len(names):
            log_metrics[k] += [0] * (len(names) - len(values))
        assert len(values) == len(names), "logger : {} ,len_data({}) : len_columns({})".format(
            k, len(values), len(names)
        )

        print_string = ""
        for name, value in zip(names, values):
            if isinstance(value, float) or isinstance(value, np.float32):
                print_string += "{}: {:.4f} ".format(name, value)
            else:
                print_string += "{}: {} ".format(name, value)
        medium["metrics_logger"].info(print_string)


def InterAndUnion(pred, mask, num_class):
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter

    return (area_inter, area_union)


def ChangeProcessingLabel2RGB(src_root, des_root):
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if "png" in file:
                image = cv2.imread(os.path.join(root, file), 0)
                if image.max() > 1:
                    # 保存四类彩色图像
                    image_color = np.zeros((image.shape[0], image.shape[1], 3))
                    image_color[:, :, 0][image == 3] = 255
                    image_color[:, :, 1][image == 2] = 255
                    image_color[:, :, 2][image == 1] = 255
                    cv2.imwrite(
                        os.path.join(root.replace(src_root, des_root), file), image_color,
                    )
                if image.max() == 1:
                    # 保存两类黑白图像
                    image[image > 0] = 255
                    cv2.imwrite(
                        os.path.join(root.replace(src_root, des_root), file), image_color,
                    )


def getShapes(obj):
    if isinstance(obj, list):
        temp_list = []
        for i in range(len(obj)):
            temp_list.append(getShapes(obj[i]))
        return temp_list
    else:
        return obj.shape


def merge(objs: list, data_ord: tuple = None, dim=0):
    if isinstance(objs, list):
        shape_list = []
        obj_list = []
        for i, obj in enumerate(objs):
            obj, shape = merge(obj, data_ord, dim)
            shape_list.append(shape)
            obj_list.append(obj)
        if isinstance(obj, torch.Tensor):
            obj_list = torch.cat(obj_list, dim=dim)
        elif isinstance(obj, np.ndarray):
            obj_list = np.concatenate(obj_list, axis=dim)
        return obj_list, shape_list
    else:
        shape = objs.shape
        if data_ord is not None:
            objs = objs.permute(*data_ord)
        return objs, shape


def split(objs: Union[torch.Tensor, np.ndarray], origin_shapes, dim, current_index=0):
    if isinstance(origin_shapes, list):
        feas_list = []
        for origin_shape in origin_shapes:
            fea, current_index = split(objs, origin_shape, dim, current_index)
            feas_list.append(fea)
        return feas_list, current_index
    else:
        num_slices = origin_shapes[dim]  # get D
        fea = objs[current_index : current_index + num_slices]
        return (fea, current_index + num_slices)


def isPath(
    path_str: str, check_type: str,
):
    linux_file_path_re = "/^\/(?:[^/]+\/)*[^/]+$/"
    windows_file_path_re = "/^[a-zA-Z]:\\(?:\w+\\)*\w+\.\w+$/"
    linux_dir_path_re = "/^\/(?:[^/]+\/)*$/"
    windows_dir_path_re = "/^[a-zA-Z]:\\(?:\w+\\?)*$/"
    result_linux_file = re.match(path_str, linux_file_path_re)
    result_windows_file = re.match(path_str, windows_file_path_re)
    result_linux_dir = re.match(path_str, linux_dir_path_re)
    result_windows_dir = re.match(path_str, windows_dir_path_re)

    is_file_path = result_linux_file is not None or result_windows_file is not None
    is_dir_path = result_linux_dir is not None or result_windows_dir is not None
    if check_type == "file":
        return is_file_path
    if check_type == "dir":
        return is_dir_path
    else:
        return is_file_path, is_dir_path


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


if __name__ == "__main__":
    df = None
    df = DFAdd2CSV(df, [1, 2, 3, 4, 8], ["epoch", "loss", "acc", "aa", "xx"])
    # for i in range(100):
    #     df = DFAdd2CSV(df,[i,i*2,i*3,i*4,math.sin(i)*i],['epoch','loss','acc','aa','xx'])
    # print(df.head())
    # CSV2Png(csv_path='log.csv',png_path='log.png')

    # df1 = pd.DataFrame(columns=['epoch','train_loss','val_loss'])
    # df2 = pd.DataFrame(data = [[0,1,2]],columns=['epoch','train_loss','val_loss'])
    # df3 = pd.DataFrame(data = [[4,10,10]],columns=['epoch','train_loss','val_loss'])
    # df1 = pd.concat([df1,df2,df3],ignore_index=True)
    # df1.to_csv('a.csv',index=False)
    # df1 = pd.read_csv('a.csv')
    # print(list(df1.columns))
    # print(df1['train_loss'])
    #
    #
    # plt.plot(df1['epoch'],df1['train_loss'],label='train_loss',c='r')
    # plt.plot(df1['epoch'],df1['val_loss'],label='val_loss',c='b')
    # plt.savefig('km_feat_img.png')
    # print(df1.values)
    # plt.plot(x=df1.loc[:,'epoch'].values,y=df1.loc[:,'train_loss'].values)
    # plt.plot(x=df1.loc[:,'epoch'].values,y=df1.loc[:,'val_loss'].values)

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import _Debug

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    gt[gt <= 0] = 0
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], device="cpu"):
    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def padding_in_test(img, ds_rate=16):
    D, H, W = img.shape
    padding_d = np.ceil(D / ds_rate) * ds_rate
    padding_h = np.ceil(H / ds_rate) * ds_rate
    padding_w = np.ceil(W / ds_rate) * ds_rate

    img = np.pad(
        img, ((0, 0), (0, int(padding_h) - H), (0, int(padding_w) - W),), mode="minimum",
    )
    origin_DHW = (D, H, W)

    return img, origin_DHW


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256], device="cpu"):
    image, label = (
        image.squeeze(0).cpu().detach().numpy(),
        label.squeeze(0).cpu().detach().numpy(),
    )

    image, origin_DHW = padding_in_test(image)

    _Debug.checkImageMatrix({"volume_batch": image, "label_batch": label})
    
    image = image[:, None]
    input = torch.from_numpy(image).float().to(device)

    with torch.no_grad():
        output_main, _, _, _ = net(input)
    output_main = output_main[:, :, : origin_DHW[1], : origin_DHW[2]]
    out = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
    prediction = out.cpu().detach().numpy()

    # prediction = np.zeros_like(label)
    # for ind in range(image.shape[0]):
    #     slice = image[ind, :, :]
    #     x, y = slice.shape[0], slice.shape[1]
    #     slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
    #     input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
    #     net.eval()
    #     with torch.no_grad():
    #         output_main, _, _, _ = net(input)
    #         out = torch.argmax(torch.softmax(output_main, dim=1), dim=1).squeeze(0)
    #         out = out.cpu().detach().numpy()
    #         pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    #         prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

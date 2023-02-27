import os
import torch
import torch.nn as nn
from Model.model3d import generate_model
from torchsummary import summary

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model_backbone(model_name, model_layer_num, print_or_no, freeze, n_input_channels, n_classes):
    global model
    if model_name == "resnet":
        model = generate_model(model_layer_num, n_input_channels=n_input_channels, n_classes=n_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    if print_or_no:
        print(summary(model, input_size=(1, 64, 64, 64)))

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        nn.init.xavier_normal_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)

        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
    return model


if __name__ == '__main__':
    epochs = 500
    auc_threshold = 0.6
    bs = 1
    train_num = 42
    val_num = 40
    learning_rate = 1e-4
    user = "hanxv"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weight_path = "./resnet50_23_new.pth"

    use_pretrain = True
    test_dataloader = False
    print_or_no = True   ###  whether to print the model par of the model
    freeze = False
    random_seed = 11

    model_name = "resnet"
    model_layer_num = 50
    model_type = 1
    n_input_channels = 1
    n_classes = 2

    model = create_model_backbone(model_name=model_name, model_layer_num=model_layer_num, print_or_no=print_or_no,
                                  freeze=freeze, n_input_channels=n_input_channels, n_classes=n_classes)
    print(summary(model, (1, 64, 64, 64)))

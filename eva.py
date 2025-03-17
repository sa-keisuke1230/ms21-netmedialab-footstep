import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from torchvision import models
from torchinfo import summary
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
import pandas as pd
from openpyxl import load_workbook

class FOOTSTEPS(Dataset):
    def __init__(self, df):
        self.data = df["x"]
        self.labels = df["y"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]
# mfcc
DIR_FOOTSTEP_LAB_K = "./npz_mfcc_cnn/footstep-lab-k_aug-true_step-one_"
DIR_FOOTSTEP_LAB_S = "./npz_mfcc_cnn/footstep-lab-s_aug-true_step-one_"
DIR_FOOTSTEP_WOOD_K = "./npz_mfcc_cnn/footstep-wood-k_aug-true_step-one_"
DIR_FOOTSTEP_WOOD_S = "./npz_mfcc_cnn/footstep-wood-s_aug-true_step-one_"
DIR_FOOTSTEP_STONE_BIG_S = "./npz_mfcc_cnn/footstep-stone_big-s_aug-true_step-one_"
DIR_FOOTSTEP_STONE_BIG_K = "./npz_mfcc_cnn/footstep-stone_big-k_aug-true_step-one_"
DIR_FOOTSTEP_STONE_SAMLL_K = "./npz_mfcc_cnn/footstep-stone_small-k_aug-true_step-one_"
DIR_FOOTSTEP_STONE_SAMLL_S = "./npz_mfcc_cnn/footstep-stone_small-s_aug-true_step-one_"
CORRIDOR_K = "./npz_mfcc_corridor_cnn/footstep-corridor-k_aug-true_step-one_"
CORRIDOR_S = "./npz_mfcc_corridor_cnn/footstep-corridor-s_aug-true_step-one_"


calc_list = [
    #{"const": DIR_FOOTSTEP_LAB_K, "model_name": "./model_mfcc/footstep-lab-k.pth", "floor_type": "lab", "shoes_type": "k", "floor_type_ja": "フロアパネル",
    # "shoes_type_ja": "革靴/ヒール"},
    #{"const": DIR_FOOTSTEP_LAB_S, "model_name": "./model_mfcc/footstep-lab-s.pth", "floor_type": "lab", "shoes_type": "s", "floor_type_ja": "フロアパネル",
    # "shoes_type_ja": "スニーカー"},
    #{"const": DIR_FOOTSTEP_WOOD_K, "model_name": "./model_mfcc/footstep-wood-k.pth", "floor_type": "wood", "shoes_type": "k", "floor_type_ja": "木材",
    # "shoes_type_ja": "革靴/ヒール"},
    #{"const": DIR_FOOTSTEP_WOOD_S, "model_name": "./model_mfcc/footstep-wood-s.pth", "floor_type": "wood", "shoes_type": "s", "floor_type_ja": "木材",
    # "shoes_type_ja": "スニーカー"},
    #{"const": DIR_FOOTSTEP_STONE_BIG_S, "model_name": "./model_mfcc/footstep-stone_big-s.pth", "floor_type": "stone_big", "shoes_type": "s", "floor_type_ja": "砂利大",
    # "shoes_type_ja": "スニーカー"},
    #{"const": DIR_FOOTSTEP_STONE_BIG_K, "model_name": "./model_mfcc/footstep-stone_big-k.pth", "floor_type": "stone_big", "shoes_type": "k", "floor_type_ja": "砂利大",
    # "shoes_type_ja": "革靴/ヒール"},
    #{"const": DIR_FOOTSTEP_STONE_SAMLL_K, "model_name": "./model_mfcc/footstep-stone_small-k.pth", "floor_type": "stone_small", "shoes_type": "k", "floor_type_ja": "砂利小",
    # "shoes_type_ja": "革靴/ヒール"},
    #{"const": DIR_FOOTSTEP_STONE_SAMLL_S, "model_name": "./model_mfcc/footstep-stone_small-s.pth", "floor_type": "stone_small", "shoes_type": "s", "floor_type_ja": "砂利小",
    # "shoes_type_ja": "スニーカー"},
    #{"const": CORRIDOR_S, "model_name": "./corridor_cnn_result/footstep-corridor-s.pth", "floor_type": "corridor", "shoes_type": "s", "floor_type_ja": "廊下",
    # "shoes_type_ja": "スニーカー"},
    {"const": CORRIDOR_K, "model_name": "./corridor_cnn_result/footstep-corridor-k.pth",  "floor_type": "corridor", "shoes_type": "k", "floor_type_ja": "廊下",
     "shoes_type_ja": "革靴/ヒール"},
]

for learn_list in calc_list:
    test_path = f"{learn_list['const']}test-data.npz"
    print("test_path:" + test_path)
    print("model_path:" + learn_list['model_name'])

    test_data = np.load(test_path)


    test_dataset = FOOTSTEPS(test_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    resnet_model = resnet34(weights=models.ResNet34_Weights.DEFAULT)
    resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU: " + str(torch.cuda.is_available()))
    resnet_model = resnet_model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet_model.parameters(), lr=2e-3)
    resnet_model.load_state_dict(torch.load(f"{learn_list['model_name']}", weights_only=True))
    #print(resnet_model)
    resnet_model.eval()
    #test_losses = 0
    actual_list, predict_list = [], []
    for data in test_loader:
        with torch.no_grad():
            x, y = data
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            x = x.unsqueeze(1)
            out = resnet_model(x)
            _, y_pred = torch.max(out, 1)
            actual_list.extend(y.cpu().numpy())
            predict_list.extend(y_pred.cpu().numpy())


    cm = confusion_matrix(actual_list, predict_list)

    # 混同行列をヒートマップで表示
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'], annot_kws={'fontsize': 13})
    plt.xlabel('予測値', fontname="MS Gothic", fontsize=18)
    plt.ylabel('正解値', fontname="MS Gothic", fontsize=18)
    plt.title(f"{learn_list['floor_type_ja']} - {learn_list['shoes_type_ja']}", fontname="MS Gothic", fontsize=25)
    plt.savefig(f"./corridor_cnn_result/mfcc-cnn_{learn_list['floor_type']}-{learn_list['shoes_type']}.jpg")
    #plt.show()



    res = classification_report(np.array(actual_list), np.array(predict_list),
                                target_names=['0', '1', '2'])


    print(res)

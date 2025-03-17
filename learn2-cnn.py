import os
import numpy as np
import torch
import torchvision.models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa

print(torch.__version__)
print(torch.cuda.is_available())
class FOOTSTEPS(Dataset):
    def __init__(self, df):
        self.data = df["x"]
        self.labels = df["y"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

def save_model(model, model_dir, save_name="model.pth"):
    path = os.path.join(model_dir, save_name)
    torch.save(model.cpu().state_dict(), path)

def train():
    DIR_FOOTSTEP_LAB_K = "./npz_mfcc_cnn/footstep-lab-k_aug-true_step-one_"
    DIR_FOOTSTEP_LAB_S = "./npz_mfcc_cnn/footstep-lab-s_aug-true_step-one_"
    DIR_FOOTSTEP_WOOD_K = "./npz_mfcc_cnn/footstep-wood-k_aug-true_step-one_"
    DIR_FOOTSTEP_WOOD_S = "./npz_mfcc_cnn/footstep-wood-s_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_BIG_S = "./npz_mfcc_cnn/footstep-stone_big-s_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_BIG_K = "./npz_mfcc_cnn/footstep-stone_big-k_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_SAMLL_K = "./npz_mfcc_cnn/footstep-stone_small-k_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_SAMLL_S = "./npz_mfcc_cnn/footstep-stone_small-s_aug-true_step-one_"

    calc_list = [
        {"const": DIR_FOOTSTEP_LAB_K, "floor_type": "lab", "shoes_type": "k"},
        {"const": DIR_FOOTSTEP_LAB_S, "floor_type": "lab", "shoes_type": "s"},
        {"const": DIR_FOOTSTEP_WOOD_K, "floor_type": "wood", "shoes_type": "k"},
        {"const": DIR_FOOTSTEP_WOOD_S, "floor_type": "wood", "shoes_type": "s"},
        {"const": DIR_FOOTSTEP_STONE_BIG_S, "floor_type": "stone_big", "shoes_type": "s"},
        {"const": DIR_FOOTSTEP_STONE_BIG_K, "floor_type": "stone_big", "shoes_type": "k"},
        {"const": DIR_FOOTSTEP_STONE_SAMLL_K, "floor_type": "stone_small", "shoes_type": "k"},
        {"const": DIR_FOOTSTEP_STONE_SAMLL_S, "floor_type": "stone_small", "shoes_type": "s"},
    ]

    for learn_list in calc_list:
        print(f"learning {learn_list['const']}")
        train_path = f"{learn_list['const']}train-data.npz"
        test_path = f"{learn_list['const']}test-data.npz"

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        train_dataset = FOOTSTEPS(train_data)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_dataset = FOOTSTEPS(test_data)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        for x, y in train_loader:
            #librosa.display.specshow(x[0].cpu().numpy(), x_axis='time', sr=44100)
            #plt.show()
            break

        # resnet34
        resnet_model = resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 全結合
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, 7)
        # GPUの設定
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("GPU: " + str(torch.cuda.is_available()))
        resnet_model = resnet_model.to(device)
        # 交差エントロピー誤差(損失関数)の定義
        loss_function = nn.CrossEntropyLoss()
        # 学習方法と学習率の設定 https://qiita.com/hibit/items/f32930dcf3d8ac5889cc
        optimizer = optim.Adam(resnet_model.parameters(), lr=2e-3)

        accuracy = 0
        global test_losses
        global train_losses
        for epoch in tqdm(range(50)):
            resnet_model.train()
            train_losses = 0
            for data in train_loader:
                optimizer.zero_grad()
                x, y = data
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.int64)
                x = x.unsqueeze(1)
                # 伝搬
                out = resnet_model(x)
                # 損失関数(交差エントロピー誤差)
                loss = loss_function(out, y)
                # 逆伝播
                loss.backward()
                optimizer.step()
                train_losses += loss.item()

            test_losses = 0
            actual_list, predict_list = [], []
            resnet_model.eval()
            for data in test_loader:
                with torch.no_grad():
                    x, y = data
                    x = x.to(device, dtype=torch.float32)
                    y = y.to(device, dtype=torch.int64)
                    x = x.unsqueeze(1)
                    out = resnet_model(x)
                    loss = loss_function(out, y)
                    _, y_pred = torch.max(out, 1)
                    test_losses += loss.item()

                    actual_list.append(y.cpu().numpy())
                    predict_list.append(y_pred.cpu().numpy())

            actual_list = np.concatenate(actual_list)
            predict_list = np.concatenate(predict_list)
            accuracy = np.mean(actual_list == predict_list)

            #print("epoch", epoch, "\t train_loss", train_losses, "\t test_loss", test_losses, "\t accuracy", accuracy)
        print("\t train_loss", train_losses, "\t test_loss", test_losses, "\t accuracy", accuracy)
        save_model(resnet_model, "./model_mfcc", save_name=f"footstep-{learn_list['floor_type']}-{learn_list['shoes_type']}.pth")

if __name__ == "__main__":
    train()
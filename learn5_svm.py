import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
import seaborn as sns

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
    CORRIDOR_K = "./npz_mfcc_corridor_svm/footstep-corridor-k_aug-true_step-one_"
    CORRIDOR_S = "./npz_mfcc_corridor_svm/footstep-corridor-s_aug-true_step-one_"

    DIR_FOOTSTEP_LAB_K = "./npz-mfcc/footstep-lab-k_aug-true_step-one_"
    DIR_FOOTSTEP_LAB_S = "./npz-mfcc/footstep-lab-s_aug-true_step-one_"
    DIR_FOOTSTEP_WOOD_K = "./npz-mfcc/footstep-wood-k_aug-true_step-one_"
    DIR_FOOTSTEP_WOOD_S = "./npz-mfcc/footstep-wood-s_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_BIG_S = "./npz-mfcc/footstep-stone_big-s_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_BIG_K = "./npz-mfcc/footstep-stone_big-k_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_SAMLL_K = "./npz-mfcc/footstep-stone_small-k_aug-true_step-one_"
    DIR_FOOTSTEP_STONE_SAMLL_S = "./npz-mfcc/footstep-stone_small-s_aug-true_step-one_"

    calc_list = [
        #{"const": DIR_FOOTSTEP_LAB_K, "floor_type": "lab", "shoes_type": "k", "floor_type_ja": "フロアパネル", "shoes_type_ja": "革靴/ヒール"},
        #{"const": DIR_FOOTSTEP_LAB_S, "floor_type": "lab", "shoes_type": "s", "floor_type_ja": "フロアパネル", "shoes_type_ja": "スニーカー"},
        #{"const": DIR_FOOTSTEP_WOOD_K, "floor_type": "wood", "shoes_type": "k", "floor_type_ja": "木材", "shoes_type_ja": "革靴/ヒール"},
        #{"const": DIR_FOOTSTEP_WOOD_S, "floor_type": "wood", "shoes_type": "s", "floor_type_ja": "木材", "shoes_type_ja": "スニーカー"},
        #{"const": DIR_FOOTSTEP_STONE_BIG_S, "floor_type": "stone_big", "shoes_type": "s", "floor_type_ja": "砂利大", "shoes_type_ja": "スニーカー"},
        #{"const": DIR_FOOTSTEP_STONE_BIG_K, "floor_type": "stone_big", "shoes_type": "k", "floor_type_ja": "砂利大", "shoes_type_ja": "革靴/ヒール"},
        #{"const": DIR_FOOTSTEP_STONE_SAMLL_K, "floor_type": "stone_small", "shoes_type": "k", "floor_type_ja": "砂利小", "shoes_type_ja": "革靴/ヒール"},
        #{"const": DIR_FOOTSTEP_STONE_SAMLL_S, "floor_type": "stone_small", "shoes_type": "s", "floor_type_ja": "砂利小", "shoes_type_ja": "スニーカー"},
        {"const": CORRIDOR_S, "floor_type": "corridor", "shoes_type": "s", "floor_type_ja": "廊下",
         "shoes_type_ja": "スニーカー"},
        {"const": CORRIDOR_K, "floor_type": "corridor", "shoes_type": "k", "floor_type_ja": "廊下",
         "shoes_type_ja": "革靴/ヒール"},
    ]

    for learn_list in calc_list:
        print(f"learning {learn_list['const']}")
        train_path = f"{learn_list['const']}train-data.npz"
        test_path = f"{learn_list['const']}test-data.npz"

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        x_train = train_data["x"]
        x_test = test_data["x"]
        y_train = train_data["y"]
        y_test = test_data["y"]

        print(len(x_test))
        print(len(y_test))

        params = [
            {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
            {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]}
        ]
        print(x_train)
        clf = GridSearchCV(svm.SVC(), params, n_jobs=-1, cv=3)
        clf.fit(x_train, y_train)
        print("学習モデル=", clf.best_estimator_)

        # 検証用データで精度を確認
        pre = clf.predict(x_test)
        ac_score = metrics.accuracy_score(pre, y_test)
        print("正解率=", ac_score)

        print(classification_report(np.array(y_test), np.array(pre),
                                    target_names=['0', '1', '2', '3']))

        cm = confusion_matrix(np.array(y_test), np.array(pre))
        # 混同行列をヒートマップで表示
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1', '2', '3'], yticklabels=['0', '1', '2', '3'], annot_kws={'fontsize': 13})
        plt.xlabel('予測値', fontname="MS Gothic", fontsize=18)
        plt.ylabel('正解値', fontname="MS Gothic", fontsize=18)
        plt.title(f"{learn_list['floor_type_ja']} - {learn_list['shoes_type_ja']}", fontname="MS Gothic", fontsize=25)
        #plt.show()
        plt.savefig(f"./corridor_svm_result/{learn_list['floor_type']}-{learn_list['shoes_type']}.jpg")


if __name__ == "__main__":
    train()
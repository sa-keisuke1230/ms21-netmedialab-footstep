# 音響による歩容認証における床素材の認証精度への影響 <br> <span style="font-size:15px">Accuracy of Acoustic Walk Through Certification by Floor Material</span>

## ファイル構成

**github repository / https://github.com/sa-keisuke1230/ms21-networkmedialab-footstep/**

| filename         | note                          |
| ---------------- | ----------------------------- |
| /\*.csv          | ラベル付のためのカテゴリー    |
| edit_mfcc.py     | CNN に入力する前処理          |
| edit_svm.py      | SVM に入力する前処理          |
| eva.py           | CNN 推論用                    |
| learn4_cnn.py    | CNN 学習と評価                |
| learn5_svm.py    | SVM 学習と評価                |
| requirements.txt | Python モジュール用(不足あり) |

**google drive/\* https://drive.google.com/drive/folders/1EZxyNw3Td9dzeVV_Ha0pDHmXdWloTCxk?usp=drive_link/drive_data.zip**

| filename              | note                                    |
| --------------------- | --------------------------------------- |
| npz_mfcc_corridor_svm | 廊下の SVM データセット                 |
| npz_mfcc_corridor_cnn | 廊下の CNN データセット                 |
| npz-mfcc              | 木材,砂利,フロアパネル SVM データセット |
| npz_mfcc_cnn          | 木材,砂利,フロアパネル CNN データセット |
| model_liner_7         | 木材,砂利,フロアパネル CNN 学習済モデル |
| model_cnn_corrider    | 廊下 CNN 学習済モデル                   |

**google drive/data/\* https://drive.google.com/drive/folders/1EZxyNw3Td9dzeVV_Ha0pDHmXdWloTCxk?usp=drive_link/drive_data.zip**

<p>歩行音の音声データをまとめたディレクトリ<br>
lab/footstep_lab_k/ : 床材: フロアパネル 履物: 革靴またはヒールの歩行音<br>
lab/footstep_lab_k/(name)_k : nameの革靴の歩行音<br>
lab/footstep_lab_s/(name)_s : nameのスニーカーの歩行音<br></p>

| filename         | note                                     |
| ---------------- | ---------------------------------------- |
| lab              | フロアパネル(研究室床)                   |
| lab_corridor     | 研究室廊下                               |
| stone            | 砂利道(大:big 小:small)                  |
| wood             | 木材                                     |
| インターホン.mp3 | インターホンを通して歩行音を録音したもの |

**google drive/result/\* https://drive.google.com/drive/folders/1EZxyNw3Td9dzeVV_Ha0pDHmXdWloTCxk?usp=drive_link/drive_data.zip**

| filename            | note                                  |
| ------------------- | ------------------------------------- |
| img                 | 履物の画像                            |
| corridor_cnn_result | 廊下 CNN 混同行列                     |
| corridor_svm_result | 廊下 SVM 混同行列                     |
| result_cnn_mfcc     | 木材,フロアパネル,砂利道 CNN 混同行列 |
| result_svm_mfcc     | 木材,フロアパネル,砂利道 SVM 混同行列 |

# load module
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import os
from tqdm import tqdm
from sklearn import model_selection
import csv
import matplotlib
matplotlib.use('TkAgg')

print("load module done")


class Footstep_Audio():
    def __init__(self, audio, sr=44100):
        '''
            Summary:
               音声波形から足音波形を取り出し、加工するクラス

            Args:
                audio_file_path: 読み込む音声波形のファイルパス
                sr(44100): サンプリング数周波数

            Retruns:
                void
        '''
        if isinstance(audio, str):
            self._y, self.sr = librosa.load(audio, sr=sr)
            self.timeArray = np.arange(0, len(self._y)) / self.sr
        elif isinstance(audio, np.ndarray):
            self._y = audio
            self.sr = sr
        else:
            raise TypeError("Invaild data type audio")

    '''
    def getNearestValue(self, list, num):
        """
            Sumarry:
                リストからある値に最も近い値を返す
            Retruns:
                対象値に最も近い値
        """
        # リスト要素と対象値の差分を計算し最小値のインデックスを取得
        idx = np.abs(np.asarray(list) - num).argmin()
        return idx

    def getTimeMS(self):
        return (len(self._y) / self.sr) * 1000

    def f(self, t):
        return self._y[t]
    '''

    def getThreshold(self):
        '''
            https://ipsj-ixsq-nii-ac-jp.teulib.idm.oclc.org/ej/index.php?action=pages_view_main&active_action=repository_action_common_download&item_id=200833&item_no=1&attribute_id=1&file_no=1&page_id=13&block_id=8
            Summary:
               二乗平均平方根
            Retruns:
                実効値 Xrms (float)
        '''
        # 音声波形 秒に変換
        # T = self.getTimeMS() / 1000
        # x(t)^2
        x_squared = self._y ** 2
        ave = np.mean(x_squared)
        return float(np.sqrt(ave) * 2)

    def calculate_melsp(self, n_fft=1024, hop_length=128):
        '''
            Summary:
               音声波形をメル周波数スペクトログラムに変換する

            Args:
                audio_file_path: 読み込む音声波形
                sr(44100): サンプリング数周波数

            Retruns:
                np.ndarray メル周波数スペクトログラム
        '''
        x = self._y
        stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length)) ** 2
        log_stft = librosa.power_to_db(stft)
        melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
        return melsp

    def show_melsp(self, melsp, fs=44100):
        librosa.display.specshow(melsp, sr=fs)
        plt.colorbar()
        plt.show()

    def add_white_noise(self, rate=0.002):
        w = self._y + rate * np.random.random(len(self._y))
        return Footstep_Audio(audio=w)

    def shift_sound(self, rate=2):
        x = self._y
        w = np.roll(x, int(len(x) // rate))
        return Footstep_Audio(audio=w)

    def stretch_sound(self, rate=1.1):
        x = self._y
        input_length = len(x)
        x = librosa.effects.time_stretch(x, rate=rate)
        if len(x) > input_length:
            return Footstep_Audio(audio=x[:input_length])
        else:
            w = np.pad(x, (0, max(0, input_length - len(x))), "constant")
            return Footstep_Audio(audio=w)

    def save_wav(self, save_name):
        '''
            Summary:
               音声波形をwav形式に書き出し

            Args:
                save_name: 保存名

            Retruns:
                void
        '''
        sf.write(str(save_name) + '.wav', self._y, self.sr, subtype='PCM_24')
        return

    def show_wave(self, lw=0.5, hlines=[], hline_color="red", hline_style="dashed", axv_y1=[], axv_y2=[],
                  axvspan_color="coral", display_size=(0, 0)):
        '''
            Summary:
               音声波形を描画

            Args:
                lw: 線の太さ
                hlines: x軸上に描画する線
                hline_color: hlinesの線の色
                hline_style: 線の種類
                axv_y1: 塗りつぶす開始座標
                axv_y2: 塗りつぶす終了座標
                axvspan_color: 塗りつぶす色
                display_size: 描画する時間範囲 (s)
                    display_size[0]: 描画開始時間(s)
                    display_size[1]: 描画終了時間(s)

            Retruns:
                void
        '''

        # 範囲限定
        _y = self._y[int(self.sr * display_size[0]):int(self.sr * display_size[1])] if display_size[0] != display_size[
            1] else self._y

        if len(axv_y1) != len(axv_y2):
            raise TypeError("axv_y1とaxv_y2の配列要素数が異なります")

        x = np.arange(0, len(_y)) / self.sr
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(111)

        if len(hlines) > -1:
            ax.hlines(hlines, min(x), max(x), hline_color, linestyles=hline_style, lw=lw)

        ax.plot(x, _y, lw=0.6)

        for i in range(len(axv_y1)):
            if axv_y1[i] > max(x):
                break

            ax.axvspan(axv_y1[i], axv_y2[i], color=axvspan_color)
            continue

        plt.show()

    def cut_n_step(self, step=1, threshold=0, prev_buff=10, zero_padding=True):
        '''
            Summary:
                音声配列から足音を切り出し、パッディングを行う

            Args:
                step: 足音を切り出す数
                threshold: 閾値
                preb_buff(ms): 閾値を超えた位置から何秒前から切り出しを行うか

            Retruns:
                {"data": Footstep_Audio[], "time": np.ndarray[]}
        '''
        _y = self._y
        sr = self.sr
        IsOverThreshold = False
        sampling_100ms = int(sr / 10)
        sampling_prevbuff_ms = int(sr / 1000) * prev_buff
        time = np.arange(0, len(_y)) / sr
        t_elapsed = sampling_100ms
        step_count = 0
        cut_begin = 0
        cut_step = []
        cut_step_time = []
        threshold = self.getThreshold() if threshold == 0 else threshold
        # print("threshold: " + str(threshold))
        for i, y in enumerate(_y):
            if float(y) > threshold or float(y) <= -threshold:
                if not IsOverThreshold:
                    IsOverThreshold = True
                    if step_count == 0:
                        cut_begin = i
                else:
                    t_elapsed = sampling_100ms
            elif IsOverThreshold:
                if t_elapsed < 1:
                    IsOverThreshold = False
                    if step_count >= step - 1:
                        step_count = 0
                        t_elapsed = sampling_100ms + sampling_prevbuff_ms
                        if (len(_y[cut_begin - sampling_prevbuff_ms:i]) > 0):
                            cut_step.append(Footstep_Audio(_y[cut_begin - sampling_prevbuff_ms:i]))
                            cut_step_time.append(time[cut_begin - sampling_prevbuff_ms:i])
                    else:
                        t_elapsed = sampling_100ms
                        step_count += 1
                else:
                    t_elapsed -= 1
        # print(f"footstep: {len(cut_step)}")
        # return {"data": cut_step, "time": cut_step_time, "threshold": threshold}

        return Cutstep_Data(cut_step, time, threshold)

class Cutstep_Data:
    '''
        Summary:
            歩数切り出し用のデータ定義,加工クラス
        Args:
            data: Footstep_Audio[]
    '''

    def __init__(self, data, time, threshold):
        self.cutstep_array = data
        self.time = time
        self.threshold = threshold

class S3:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    def exists_dir(self, dir):
        res = self.s3_client.list_objects(Bucket=self.bucket_name, Prefix=dir)
        return "Contents" in res

    def create_dir(self, dir):
        if not self.exists_dir(dir):
            # ！パスの末尾がスラッシュでない場合ディレクトが作成されない…
            self.s3_client.put_object(Bucket=self.bucket_name, Key=dir)

    def upload_file(self, upload_file, upload_path, save_name):
        self.s3_client.upload_file(
            Filename=upload_file, Bucket=self.bucket_name, Key=os.path.join(upload_path, save_name))

    def getObject(self, key):
        return self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=key
        )

class Footstep_Data:
    def __init__(self, data, path, file_name, category):
        self.data = data
        self.path = path
        self.file_name = file_name
        self.category = category

class FootstepAugmentationData(Footstep_Data):
    def __init__(self, footstep_data, white_noise, stretch, shift):
        super().__init__(data=footstep_data.data, path=footstep_data.path, file_name=footstep_data.file_name,
                         category=footstep_data.category)
        self.white_noise = white_noise
        self.stretch = stretch
        self.shift = shift

    def show_all_allwaves(self):
        size = (2, 2)
        fig, ax = plt.subplots(size[1], size[0], figsize=(15, 5))
        fig.canvas.draw()
        fig.canvas.flush_events()
        ax[0, 0].plot(self.data._y)
        ax[0, 0].set_title("row")
        ax[0, 1].plot(self.white_noise._y)
        ax[0, 1].set_title("white noise")
        ax[1, 0].plot(self.stretch._y)
        ax[1, 0].set_title("stretch")
        ax[1, 1].plot(self.shift._y)
        ax[1, 1].set_title("shift")
        fig.tight_layout()
        plt.show()

class FootstepDataLoader():
    def __init__(self, train_dataloader, test_dataloader):
        self.train_DataLoader = train_dataloader
        self.test_DataLoader = test_dataloader

class FOOTSTEP_DATASET():
    '''
        Summary:
            足音データセットクラス
        Args:
            audio_path: 読み込む音声ファイルパス
            data_augmentation: 水増し
        Retruns:
    '''

    def __init__(self, audio_path, step_cut=1, augmentation=True):
        print("loading audio file")
        # 初期化
        label_file = "./category_corridor.csv"
        footsteps = []
        self.augmentation = augmentation
        self.category = {}

        # ラベル設定読み込み
        with open(label_file, mode="r") as f:
            reader = list(csv.reader(f))
            for i in range(len(reader[0])):
                self.category[reader[1][i].replace(" ", "")] = int(reader[0][i])

        for i, audio_file in enumerate(tqdm(audio_path)):
            # 音声データ読み込み
            data = Footstep_Audio(audio=audio_file, sr=44100)
            name = os.path.basename(audio_file).split("_")[0]
            if name not in ["egawa", "harada", "hoshi", "kasai", "kushida", "sakuma", "taguti", "aruga", "terasawa", "saito"]:
                continue
            if step_cut > 0:
                # 足音切り取り
                cut = data.cut_n_step(step=step_cut)  # : Cutstep_Data
                for c in cut.cutstep_array:
                    footsteps.append(
                        Footstep_Data(
                            data=Footstep_Audio(audio=c._y, sr=44100),
                            path=audio_file,
                            file_name=os.path.basename(audio_file),
                            category=self.category[name]
                        )
                    )

                    continue
            else:
                # 切り出しなし
                footsteps.append(
                    Footstep_Data(
                        data=data,
                        path=audio_file,
                        file_name=os.path.basename(audio_file),
                        category=self.category[name]
                    )
                )

            continue

        self.footsteps = footsteps
        # zero padding
        fd = [x.data._y for x in self.footsteps]
        max_i = self.getLenMaxIndex(fd)

        for i, fd in enumerate(self.footsteps):
            if i == max_i:
                continue
            diff = len(self.footsteps[max_i].data._y) - len(fd.data._y)
            fd.data._y = np.pad(fd.data._y, pad_width=(0, diff), mode="constant")

            continue

        # augmentaion
        if augmentation:
            self.aug = self.data_augmentation()

        self.result_data = []
        self.label = []

        if self.augmentation:
            # 水増しあり
            for i in self.aug:
                category = i.category
                whitenoise = i.white_noise._y
                stretch = i.stretch._y
                shift = i.shift._y
                raw = i.data._y

                self.result_data.extend([raw, shift, stretch, whitenoise])
                self.label.extend([category] * 4)
        else:
            # 水増しなし
            for i in self.footsteps:
                self.result_data.extend([i.data])
                self.label.extend([i.category])

        print("melsp")
        for i, d in enumerate(tqdm(self.result_data)):
            #feature_melsp = librosa.feature.melspectrogram(y=d, sr=44100)
            #feature_melsp_db = librosa.power_to_db(feature_melsp, ref=np.max)
            mfcc = librosa.feature.mfcc(y=d, n_mfcc=20, sr=44100)[1:]

            # mfcc = np.average(mfcc, axis=1)
            #librosa.display.specshow(mfcc, x_axis='time', sr=44100)
            #plt.show()
            #mfcc = mfcc.flatten()
            #mfcc = mfcc.tolist()
            self.result_data[i] = mfcc

        print(f"データセット作成完了")
        print(f"データ総数: {len(self.result_data)}")
        print(f"ラベル総数: {len(self.label)}")
        print(self.label)
        return

    def getLenMaxIndex(self, arr):
        res = 0
        len_max = 0
        for i, a in enumerate(arr):
            if len_max < len(a):
                len_max = len(a)
                res = i
        return res

    def save_npz(self, test_size=0.25, batch_size=32, shuffle=True, is_upload_s3=True, test_npz_save_path="./",
                 train_npz_save_path="./", test_npz_name="test_data", train_npz_name="train_data"):
        '''
            Summary:
                DataLoaderを生成
            Args:

            Retruns:
                FootstepDataLoader
            Todo:
                ・保存名の指定をできるように
        '''

        # 学習、評価用にデータとラベルを分ける
        x_train, x_test, y_train, y_test = model_selection.train_test_split(self.result_data, self.label,
                                                                            test_size=test_size)

        # データローダーの生成
        print("saving npz")
        # print(f"x_train: {x_train[0:5]}, y_train: {y_train}, x_test: {x_test[0:5]}, y_test: {y_test}")
        # print(x_train[0].shape)
        test_npz_save_path = os.path.join(test_npz_save_path, test_npz_name)
        train_npz_save_path = os.path.join(train_npz_save_path, train_npz_name)

        np.savez(train_npz_save_path, x=x_train, y=y_train)
        np.savez(test_npz_save_path, x=x_test, y=y_test)

        return

    def data_augmentation(self) -> FootstepAugmentationData:
        '''
            Summary:
                音声配列から水増しを行う
            Args:

            Retruns:
                FootstepAugmentationData: []
        '''
        data = []
        print("data augmentation")
        for y in tqdm(self.footsteps):
            y_data = y.data
            x0 = y_data
            x1 = y_data.add_white_noise()
            x2 = y_data.shift_sound()
            x3 = y_data.stretch_sound()
            data += [FootstepAugmentationData(footstep_data=y, white_noise=x1, shift=x2, stretch=x3)]
        return data

def main():
    #DIR_FOOTSTEP_LAB_K = "./data/lab/footstep_lab_k/"
    #DIR_FOOTSTEP_LAB_S = "./data/lab/footstep_lab_s/"
    #DIR_FOOTSTEP_WOOD_K = "./data/wood/footstep_wood_k/"
    #DIR_FOOTSTEP_WOOD_S = "./data/wood/footstep_wood_s/"
    #DIR_FOOTSTEP_STONE_BIG_S = "./data/stone/footstep_stone_big_s/"
    #DIR_FOOTSTEP_STONE_BIG_K = "./data/stone/footstep_stone_big_k/"
    #DIR_FOOTSTEP_STONE_SAMLL_K = "./data/stone/footstep_stone_small_k/"
    #DIR_FOOTSTEP_STONE_SAMLL_S = "./data/stone/footstep_stone_small_s/"
    DIR_LAB_CORRIDOR_S = "./data/lab_corridor/s/"
    DIR_LAB_CORRIDOR_K = "./data/lab_corridor/k/"


    calc_list = [
        #{"const": DIR_FOOTSTEP_LAB_K, "floor_type": "lab", "shoes_type": "k"},
        #{"const": DIR_FOOTSTEP_LAB_S, "floor_type": "lab", "shoes_type": "s"},
        #{"const": DIR_FOOTSTEP_WOOD_K, "floor_type": "wood", "shoes_type": "k"},
        #{"const": DIR_FOOTSTEP_WOOD_S, "floor_type": "wood", "shoes_type": "s"},
        #{"const": DIR_FOOTSTEP_WOOD_S, "floor_type": "wood", "shoes_type": "s"},
        #{"const": DIR_FOOTSTEP_STONE_BIG_S, "floor_type": "stone_big", "shoes_type": "s"},
        #{"const": DIR_FOOTSTEP_STONE_BIG_K, "floor_type": "stone_big", "shoes_type": "k"},
        #{"const": DIR_FOOTSTEP_STONE_SAMLL_K, "floor_type": "stone_small", "shoes_type": "k"},
        #{"const": DIR_FOOTSTEP_STONE_SAMLL_S, "floor_type": "stone_small", "shoes_type": "s"},
        {"const": DIR_LAB_CORRIDOR_S, "floor_type": "corridor", "shoes_type": "s"},
        #{"const": DIR_LAB_CORRIDOR_K, "floor_type": "corridor", "shoes_type": "k"},
    ]

    # 音声ファイルを読み込み、データセットを作成する
    for const in calc_list:
        audio_base_path = const["const"]
        print(f"SELECTED type: {audio_base_path}")

        audio_file_list = os.listdir(audio_base_path)

        # 相対パスに変換
        audio_path = [os.path.join(audio_base_path, x) for x in audio_file_list]

        # データセット作成
        dataset = FOOTSTEP_DATASET(audio_path=audio_path, augmentation=True, step_cut=1)

        # データを分けてDataLoader作成
        floor_type = const["floor_type"]
        shoes_type = const["shoes_type"]
        dataset.save_npz(
            test_size=0.25,
            batch_size=32,
            shuffle=True,
            is_upload_s3=False,
            test_npz_save_path="npz_mfcc_corridor_cnn",
            train_npz_save_path="npz_mfcc_corridor_cnn",
            test_npz_name=f"footstep-{floor_type}-{shoes_type}_aug-true_step-one_test-data",
            train_npz_name=f"footstep-{floor_type}-{shoes_type}_aug-true_step-one_train-data"
        )

if __name__ == "__main__":
    main()

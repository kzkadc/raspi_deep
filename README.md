# Raspberry Piとディープラーニングで画像認識やろうとしたら重かったのでサーバーに計算させた
[Qiitaに書いた記事](https://qiita.com/kzkadc/items/bcf817704549295875ee)のコードです。

## 準備
### サーバー側
Chainer, NumPy, Pillowを使います。
```bash
sudo pip install numpy chainer pillow
```

また，`synset_words.txt`を[こちら](https://github.com/leetenki/googlenet_chainer)からダウンロードして同じフォルダに配置します。

### クライアント（ラズパイ）側
NumPyとOpenCVを使います。

NumPyのインストール：
```bash
sudo pip install numpy
```

OpenCVは[こちらの記事](https://qiita.com/mt08/items/e8e8e728cf106ac83218)を参考にインストールします。

## 実行
サーバー側で`vgg_server.py`を起動した状態で，ラズパイ側で`vgg_client.py`を実行します。

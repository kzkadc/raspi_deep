# coding: utf-8
import socket, threading
import chainer
import chainer.links as L
import numpy as np

class ImageNetPredictor:
    def __init__(self):
        self.model = L.VGG16Layers()    # 初回実行時はモデルをダウンロードするので時間がかかる
        self.categories = np.loadtxt("synset_words.txt", str, delimiter="\n").tolist()

    def __call__(self, x):
        x = x[:,:,::-1] # BGR -> RGB
        h = self.model.extract([x], layers=["fc8"])["fc8"]
        h = h.array.argmax()    # 出力は1000次元で各カテゴリのスコアを表すので最大値のインデックスを求める

        return self.categories[h]

class Handler:
    def __init__(self, model):
        self.model = model

    def __call__(self, clientsock, client_address):
        data = b""
        while True:
            r = clientsock.recv(2048)   # 分割して受信
            data += r

            if len(data) >= 224*224*3:
                # 画像サイズ224*224*3のバイト数だけ受信したらループを抜ける
                break

        data = np.fromstring(data, dtype=np.uint8)
        data = data.reshape((224,224,3))    # データが1次元配列になってしまっているので整形する

        category = self.model(data) # 画像認識
        print(" ", client_address, category)
        clientsock.sendall(category.encode("utf-8"))    # 認識結果（カテゴリ名の文字列）をクライアントに返す

        clientsock.close()  # ソケットを閉じる

def main():
    HOST, PORT = "", 55555

    serversock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    serversock.bind((HOST, PORT))
    serversock.listen(20)

    model = ImageNetPredictor() # モデル
    print("OK")

    while True:
        clientsock, client_address = serversock.accept() #接続されればデータを格納
        print("conected to "+client_address[0])

        # 接続されたら新規にスレッドを立てて処理する
        handle_thread = threading.Thread(target=Handler(model), args=(clientsock, client_address), daemon=True)
        handle_thread.start()

if __name__ == "__main__":
    main()
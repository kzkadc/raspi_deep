# coding: utf-8
import socket, time
import numpy as np
import cv2

# サーバーのIPアドレス（適宜変える），ポート番号
HOST, PORT = "192.168.31.150", 55555

def image_recog(img):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientsock:
        clientsock.connect((HOST, PORT))
        clientsock.send(img.tostring())
        data = clientsock.recv(2048)

    return data.decode("utf-8")

img = cv2.imread("pizza.jpg")
img = cv2.resize(img, (224,224))

print("send image")
start_time = time.time()    # 認識結果が出るまでの時間を計測
category = image_recog(img)
elapsed_time = time.time() - start_time
print(category, elapsed_time)
# coding:utf-8
import numpy

import cv2
import os
import time

def gray_transform():
    '''
    图片读取及灰度转换
    '''
    a = cv2.imread('../asset/tt.jpg', cv2.IMREAD_GRAYSCALE)  # imread会删除alpha通道，导致透明度丧失
    cv2.imwrite('../asset/a.png', a)


def random_array_2_img():
    r_a = bytearray(os.urandom(120000))
    f_r_a = numpy.array(r_a)
    gray_img = f_r_a.reshape(300, 400)
    cv2.imwrite('../asset/r.png', gray_img)


def canny_detect_img():
    a = cv2.imread('../asset/tt.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('../asset/a.png', cv2.Canny(a, 200, 300))


def sift_detect_img():
    g = cv2.cvtColor(cv2.imread('../asset/tt.jpg'), cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    k,d = sift.detectAndCompute(g,None)
    i = cv2.drawKeypoints(image=g, outImage=g, keypoints=k,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                          color=(51, 163, 236))
    cv2.imshow('sift', i)
    time.sleep(20)

def corner_detect_img():
    img = cv2.imread('../asset/tt.jpg')
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = numpy.float32(g)
    dst = cv2.cornerHarris(g, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst>0.01*dst.max()] = [0, 0, 255]
    cv2.imshow('dst', img)
    time.sleep(20)

def run():
    # gray_transform()
    # random_array_2_img()
    # canny_detect_img()
    # sift_detect_img()
    corner_detect_img()


if __name__ == "__main__":
    run()

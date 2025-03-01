import pymysql
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import laod_model

import PIL
from torchvision import transforms


# model
model = load_model('7seg_vgg16_model_v1.h5')
grayscale_tfm = transforms.Grayscale(num_output_channels=3)


# find 7seg digit location 
H_W_Ratio = 1.9
THRESHOLD = 35
arc_tan_theta = 6.0
crop_y0 = 215
crop_y1 = 470
crop_x0 = 260
crop_x1 = 890 

def load_image(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussiaBlur(gray_img, (7,7), 0)
    return blurred, gray_img

def preprocess(img, threshold, show=False, kernel_size=(5,5)):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(6,6))
    img = clahe.apply(img)
    ret, dst = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
    # dst = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, threshold)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
    return dst

def helper_extract(one_d_array, threshold=20, thxy=2):
    res = []
    flag = 0 
    temp = 0 
    for i in range(len(one_d_array)):
        if one_d_array[i] < 9000*thxy:  # original threshold : 2*255
            if flag > threshold:
                start = i-flag
                end = i
                temp = end
                if end-start > 25:  # 글자 사이 떨어진 간격 최솟값, 원래 25
                    res.append((start, end))
            flag = 0
        else: 
            flag += 1
    
    else: 
        if flag > threshold: 
            start = temp
            end = len(one_d_array)
            if end-start>50: 
                res.append((start, end))
    
    return res

def find_digits_positions(img, reserved_threshold=20): 
    digits_positions = []
    img_array = np.sum(img, axis=0) # vertical sum array 
    # print(img_array)
    # print(img_array.shape)
    horizon_position = helper_extract(img_array, threshold=reserved_threshold, thxy=2)
    img_array = np.sum(img, axis=1) # horizontal sum array
    # print(img_array)
    # print(img_array.shape)
    vertical_position = helper_extract(img_array, threshold=reserved_threshold * 4, thxy=3)
    # make vertical_position has only one element 
    if len(vertical_position) > 1:
        vertical_position = [(vertical_position[0][0], vertical_position[len(vertical_position)-1][1])]
    for h in horizon_position:
        for v in vertical_position: 
            digits_positions.append(list(zip(h,v)))
    # assert len(digits_positions) > 0, "Failed to find digits's positions"

    return digits_positions

def one_digit_from_frame(frame):
    blurred, gray_img = load_image(frame)
    output = blurred
    dst = preprocess(blurred, THRESHOLD, show=True)
    digits_positions = find_digits_positions(output)
    for c in digits_positions:
        x0, y0 = c[0]
        x1, y1 = c[1]
        roi = dst[y0:y1, x0:x1]
        h, w = roi.shape
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 0, 0), 2)
    '''
    if True:
        cv2.imshow('output', output)
        cv2.waitKey()
        cv2.destroyAllWindows()
    '''
        
    return output


# classify(recognize) 7seg digit 
def recognize_digit(digit_imgs):
    digits = []
    for digit in digit_imgs:
        digit = cv2.resize(digit, dsize=(224,224))
        img_array = np.array(digit)

        img = PIL.Image.fromarray(img_array)
        after_rgb = grayscale_tfm(img)
        img_array = np.array(after_rgb)

        test_num = img_array.reshape((1,224,224,3))

        prediction = model.predict(test_num)
        pred = prediction[0].tolist()
        print(np.array(prediction[0]))
        print(pred.index(max(pred)))
        digits.append(pred.index(max(pred)))

    return digits 


# send 7seg digit result to database 
def dbconnect():
    conn = pymysql.connect(host='52.79.144.109', user='root', password='dahyun617', db='washer', charset='utf8')
    return conn

def insert_data(conn, time_send):
    cur = conn.cursor()
    sql = "INSERT INTO timeFromWebCam (time, id) VALUES (" + str(time_send) + ", 1);"
    with conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit() 


# main
def main():
    conn = dbconnect()
    print("DB Connection Completed")
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    start = time.time()
    while webcam.isOpened():  # loop through frames 
        status, frame = webcam.read()  # read frame from webcam 
        if not status: 
            break 

        now = time.time()
        digit_imgs = []
        digit = ''
        img_digits_detected, digit_imgs = one_digit_from_frame(frame)
        
        cv2.imshow("Real-time 7seg digit detection", img_digits_detected) # display output

        if len(digit_imgs) > 0 : 
            digits = recognize_digit(digit_imgs)
            print(digits)

            if now-start>=10 : 
                # send to DB
                for x in digits: 
                    digit += str(x)
                insert_data(conn, digit)
                start = time.time()
                digit = ''

        if cv2.waitKey(1) & OxFF == ord('q'):  # press 'Q' to stop
            break
    
    webcam.release()
    cv2.destroyAllWindows()

    conn.close()
    print("DB Connection Closed")


if __name__ == '__main__':
    main() 
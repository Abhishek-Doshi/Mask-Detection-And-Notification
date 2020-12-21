import numpy as np
import cv2
import random
from PIL import Image, ImageGrab
import pickle
from model import getFeatureVector
from model_images import cv2_to_pil
import smtplib
import time
import pandas as pd

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier('haarcascade_upperbody.xml')

bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
weared_mask_font_color = (0, 255, 0)
not_weared_mask_font_color = (0, 0, 255)
name_color = (255, 0, 0)
time_color = (255, 255, 255)
thickness = 1
font_scale = 0.5
weared_mask = "Mask Detected"
not_weared_mask = "No Mask Detected"
sender_email = ''
password = ''
server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
server.login(sender_email, password)
email_data = pd.read_csv(r'email_data.csv')
email_dict = {str(name) : False for name in email_data['name'] }

def predict_faces(faces):
    text = []
    if len(faces) > 0:
        vec_matrix = getFeatureVector(faces, matrix = True, preprocess = True, pre_compute = True)
        vec_matrix = np.squeeze(np.asarray(vec_matrix))
        if len(vec_matrix) == 128: vec_matrix = np.expand_dims(vec_matrix, axis = 0)
        text = list(loaded_model.predict(vec_matrix))
    return text

def send_email(name, tm, server = server):
    if len(email_data[email_data['name'] == str(name)]) == 1:
        if email_dict[str(name)] == False:
            message = str("Dear " + str(name) + ", you were detected not wearing a mask on " + str(tm) + " IST. Please wear a mask. Upon another detection you will be marked as default.")
            email = email_data[email_data['name'] == str(name)]['email'][0]
            server.sendmail(sender_email, email, message)
            print(message)
            email_dict[str(name)] = True

loaded_model = pickle.load(open('svm.sav', 'rb'))
cap = cv2.VideoCapture('vdo.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Mask Detection.mp4', fourcc, 20.0, (500,  300))

while cap.isOpened():

    ret, img = cap.read()
    if not ret: break

    try: 
        img = cv2.resize(img, (500, 300))      
        
        cv2.putText(img, str(time.ctime()), (240, 20), font, font_scale, time_color, thickness, cv2.LINE_AA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)
        maskless_faces = []
        maskless_faces_coordinates = []

        if(len(faces) == 0 and len(faces_bw) == 0): pass

        elif(len(faces) == 0 and len(faces_bw) == 1):
            for (x, y, w, h) in faces_bw:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, weared_mask, (x, y-5), font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                roi_color = Image.fromarray(img[y:y + h, x:x + w])
                mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
                if(len(mouth_rects) == 0):
                    cv2.putText(img, weared_mask, (x, y-10), font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
                else:
                    for (mx, my, mw, mh) in mouth_rects:
                        if(y < my < y + h):
                            maskless_faces.append(roi_color)
                            maskless_faces_coordinates.append((x, y, w, h))
                            cv2.putText(img, not_weared_mask, (x+w+5, y+12), font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
                            cv2.putText(img, 'Getting Identity...', (x+w+5, y+36), font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)

        cv2.imshow('Mask Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'): cv2.destroyAllWindows()


        if len(maskless_faces) > 0:
            text = predict_faces(maskless_faces)
            for i, (x, y, w, h) in enumerate(maskless_faces_coordinates):
                cv2.putText(img, text[i], (x+w+5, y + 60), font, font_scale + 0.2, name_color, thickness + 1, cv2.LINE_AA)
                tm = time.ctime().split(':')
                tm = tm[0][:-3] + str(',') + tm[0][-3:] + str(' hrs ')+ tm[1] + str(' mins')
                send_email(text[i], tm = tm)
                cv2.putText(img, 'Email Sent', (x+w+5, y+90), font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
        out.write(img)
        cv2.imshow('Mask Detection', img)
        if cv2.waitKey(25) & 0xFF == ord('q'): cv2.destroyAllWindows()
        
    except cv2.error: out.write(img)
    

server.quit()
cap.release()
out.release()
cv2.destroyAllWindows()

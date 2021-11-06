import cv2
import matplotlib.pyplot as plt
import numpy as np
import face_recognition
from datetime import datetime
import os
def mark_attendence(name):
    with open('attendence.csv','r+')as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in  mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dstring}')


encoding_list=[]
img_elon=face_recognition.load_image_file('../../../Downloads/elon musk.jpg')
# locations gets x,y location of face top,right,bottom left
face_elon=face_recognition.face_locations(img_elon)[0]
#encoding gets 128 measurments like distance between eye,nose etc for img comparisions
encoding_list.append(face_recognition.face_encodings(img_elon)[0])
img_billy=face_recognition.load_image_file('../../../Downloads/billy.jpg')
# locations gets x,y location of face top,right,bottom left
face_billy=face_recognition.face_locations(img_billy)[0]
#encoding gets 128 measurments like distance between eye,nose etc for img comparisions
encoding_list.append(face_recognition.face_encodings(img_billy)[0])
img_jimmy=face_recognition.load_image_file('../../../Downloads/jeffy.jpg')
# locations gets x,y location of face top,right,bottom left
face_jimmy=face_recognition.face_locations(img_jimmy)[0]
#encoding gets 128 measurments like distance between eye,nose etc for img comparisions
encoding_list.append(face_recognition.face_encodings(img_jimmy)[0])
img_utkarsh=face_recognition.load_image_file('../../../Desktop/utkarsh.jpg')
# locations gets x,y location of face top,right,bottom left
face_utkarsh=face_recognition.face_locations(img_utkarsh)[0]
#encoding gets 128 measurments like distance between eye,nose etc for img comparisions
encoding_list.append(face_recognition.face_encodings(img_utkarsh)[0])
classname=['elon musk','bill gates','jeff bezoz','utkarsh']
print('encoding complete')
cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)
    face_curr_frame=face_recognition.face_locations(imgs)
    face_curr_encoding=face_recognition.face_encodings(imgs,face_curr_frame)
    for a,b in zip(face_curr_frame,face_curr_encoding):
        matches=face_recognition.compare_faces(encoding_list,b)
        #print(matches)
        face_dis=face_recognition.face_distance(encoding_list,b)
        #print(face_dis)
        matchg_index=np.argmin(face_dis)
        if matches[matchg_index]:
            name=classname[matchg_index].upper()
            #print(name)
            y1,x2,y2,x1=a
            y1=y1*4
            x1=x1*4
            y2=y2*4
            x2=x2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            mark_attendence(name)

    cv2.imshow('Webcame',img)
    cv2.waitKey(1)


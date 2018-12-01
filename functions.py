#import classifiers.custom_classifier as classification
import classifiers.custom_classifier_tcc as classification
#import classifiers.mobileNetClassifier as classification
import numpy as np
import cv2
import requests
import datetime
import json
import os

# functions ===================================================================
def predict_next_position(pointpositionhistory):
    numofpos=len(pointpositionhistory)
    if numofpos == 1:
        predicted_next_position=1
    return predicted_next_position

def point_existed(p,points_history):
    #se distancia entre ponto e algum dos elementos de lasthull.predict for menor que 10, True
    distlist=[]
    if len(points_history)==0:
        return False
    for el in points_history:
        distlist.append(np.linalg.norm(np.subtract(p,el[len(el)-1])))
    minval=min(distlist)
    if minval > 40:
        return False
    else:
        return True
    
def get_point_pos(p,points_history):
    # returns the position of the vehicle with the last location closest to the point location
    distlist=[]
    for el in points_history:
        distlist.append(np.linalg.norm(np.subtract(p,el[-1])))
    minpos=distlist.index(min(distlist))
    return minpos

def point_crossed_line(points_history,line,car_orientation):
    if len(points_history) < 1:
        return (False,(0,0),(0,0))
    up=False
    down=False
    if car_orientation == 'vertical':
        for point_history in points_history:
            up=False
            down=False
            for point in point_history:
                if point[1] > line[0][1]:
                    up=True
                else:
                    down=True
                if up and down:
                    #points_history.remove(point_history)
                    return (True, point,point_history)
    elif car_orientation == 'horizontal':
        for point_history in points_history:
            left=False
            right=False
            for point in point_history:
                if point[0] > line[0][0]:
                    right=True
                else:
                    left=True
                if left and right:
                    #points_history.remove(point_history)
                    return (True, point,point_history)
    return (False,(0,0),(0,0))

def classify_point(point,rectangles,frame):
    print( len(rectangles))
    distlist=[];
    for rect in rectangles:
        xret=rect[0]+rect[2]/2
        yret=rect[1]+rect[3]/2
        distlist.append(np.linalg.norm(np.subtract(point,(xret,yret))))
    minpos=distlist.index(min(distlist))
    x1=rectangles[minpos][0]
    x2=rectangles[minpos][0]+rectangles[minpos][2]
    y1=rectangles[minpos][1]
    y2=rectangles[minpos][1]+rectangles[minpos][3]
    roi=frame[y1:y2,x1:x2]
    classification.classify(roi)
    
def send_to_server(point,rectangles,frame):
    distlist=[];
    for rect in rectangles:
        xret=rect[0]+rect[2]/2
        yret=rect[1]+rect[3]/2
        distlist.append(np.linalg.norm(np.subtract(point,(xret,yret))))
    minpos=distlist.index(min(distlist))
    x1=rectangles[minpos][0]
    x2=rectangles[minpos][0]+rectangles[minpos][2]
    y1=rectangles[minpos][1]
    y2=rectangles[minpos][1]+rectangles[minpos][3]
    roi=frame[y1:y2,x1:x2]
    timestamp = datetime.datetime.now()
    filename = timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f.jpg')
    print("[INFO] captured object: {}".format(filename))
    cv2.imwrite(filename, roi)
    params = {'sensor_id': 1,
              'time_stamp': timestamp.strftime('%Y-%m-%d_%H-%M-%S-%f'),
              'direction': 'l-r'}
    headers = {"Api-Key": "dead-beef-4-feed", "Accept": "*/*"}
    files = {
        'json': (None, json.dumps(params), 'application/json'),
        'file': (os.path.basename(filename),
                 open(filename, 'rb'), 'application/octet-stream')
    }
    r = requests.post("http://localhost:8000/upload", headers=headers, files=files)
    print("[HTTP-POST-ret] {}".format(r.text))
    
def drop_point(point_history,points_history):
    points_history.remove(point_history)
    
    
#==============================================================================
    
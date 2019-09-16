import multiprocessing 
import time 
import cv2

#maskrcnn put in here
def subroi(img, output) :
    bbox = output[:4]
    id = output[-1]
    area_threshold = 100000  
    area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    roiImg = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    if area > area_threshold:
        
        if id%2 == 0:
            cv2.imwrite("bdox2.jpg",roiImg)
        else : 
            cv2.imwrite("bdox1.jpg",roiImg)


def get_bbox_area(bbox):
    return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

#def box(outputs, ):

import os
import cv2
import time
import argparse
import numpy as np

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

from mtp import subroi, get_bbox_area
import multiprocessing as mp
from multiprocessing import Process
import time

people_path = []
area_dic={}

class Detector(object):
    def __init__(self, args):
        self.args = args
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        self.deepsort = DeepSort(args.deepsort_checkpoint)
        self.class_names = self.yolo3.class_names


    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 30, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        

    def detect(self):
        #multicore
        pool = mp.Pool(processes=6) #6-core
        #xmin, ymin, xmax, ymax = self.area
        jump_flag = 1 
        while self.vdo.grab(): 
            _, ori_im = self.vdo.retrieve()
            if jump_flag%2 ==0 : #jump frame  
                start = time.time()
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                im = ori_im
                bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
                if bbox_xcycwh is not None:
                    # select class person
                    mask = cls_ids==0

                    bbox_xcycwh = bbox_xcycwh[mask]
                    bbox_xcycwh[:,3:] *= 1.2

                    cls_conf = cls_conf[mask]
                    outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                    for output in outputs:
                        if output[4] > len(people_path):
                            for i in range(0, output[4] - len(people_path)):
                                people_path.append([])
                        people_path[output[4] - 1].append(np.array(([(output[0] + output[2]) / 2, output[3]])))
                        coordinate = output[:4]
                        bbox_area = get_bbox_area(coordinate)
                        try :
                            if area_dic[output[-1]] < bbox_area :
                                area_dic[output[-1]] = bbox_area
                                pool.apply_async(subroi,(ori_im,output))
                                print("---------------")

                        except KeyError:
                            area_dic.setdefault(output[-1],bbox_area)
                            pool.apply_async(subroi,(ori_im,output))
                            print("---------------")
                    
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:,:4]
                        identities = outputs[:,-1]
                        ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)
                        for id in identities:
                            for i in range(1, len(people_path[id-1])):
                                cv2.line(ori_im, (int(people_path[id-1][i-1][0]), int(people_path[id-1][i-1][1])), 
                                (int(people_path[id-1][i][0]), int(people_path[id-1][i][1])), (0, 0, 255), 1)
                end = time.time()
                print("time: {}s, fps: {}".format(end-start, 1/(end-start)))
                print(area_dic)
            jump_flag+=1
            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(ori_im)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()

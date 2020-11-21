# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 12:58:54 2020

@author: JLD
"""

from sort import *
import numpy as np
import cv2

class VehicleTracker:
    def __init__(self):
        pass
    
    
    def load_detections(self, file_name, file_format=1):
        """
        file_format = 1   ==>>  frame,x1,y1,x2,y2,score,class
        file_format = 2   ==>>  frame,x1,y1,w,h,score,class
        """
        self.detections = np.loadtxt(file_name, delimiter=',')
        
        # convert format from 2 to 1
        if file_format==2:
            self.detections[:,3:5] +=  self.detections[:,1:3]
        
        # list all classes in detections
        self.classes = []
        for vehicle_class in self.detections[:,6]:
            if vehicle_class not in self.classes:
                self.classes.append(vehicle_class)
        
        # calculate number of frames        
        self.total_frames = int(self.detections[:,0].max())
    
    
    def SORT(self, max_age=1, min_hits=3, iou_threshold=0.3):  
        self.sort_tracker = Sort(max_age,        # Maximum number of frames to keep alive a track without associated detections
                                 min_hits,       # Minimum number of associated detections before track is initialised
                                 iou_threshold)  # Minimum IOU for match
        
        
    def tracker(self, module='SORT', save=True, file_out='track.txt'):
        if module=='SORT':
            module_tracker = self.sort_tracker
            
        trackers_list = []
           
        for vehicle_class in self.classes:
            det_class = self.detections[self.detections[:,6]==vehicle_class,0:6]    # Extract class detections
            
            for frame in range(1,self.total_frames+1):
                det_frame = det_class[det_class[:,0]==frame,1:6]                    # Extract frame detections
                trackers_frame = module_tracker.update(det_frame)
        
                for linha in trackers_frame:
                    linha = np.append(vehicle_class,linha)
                    linha = np.append(frame,linha)
                    trackers_list.append(linha)
        
        sort_frame = lambda x:x[0]
        trackers_list.sort(key=sort_frame)          
        self.trackers = np.array(trackers_list)  # frame,class,x1,y1,x2,y2,id
                    
        if save:
            np.savetxt(file_out,self.trackers,delimiter=',',fmt=['%d','%d','%.4f','%.4f','%.4f','%.4f','%d'])
    
    
    def write_video(self,frame_dir, video_out, fps, video_size):
        out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc('M','J','P','G') , fps, (video_size))
        frame_numbers = int(self.trackers[:,0].max())
        for i in range(frame_numbers):
            frame = i + 1
            frame_path = frame_dir + '/frame_' + str(frame) + '.jpg'
            img = cv2.imread(frame_path)
            dets_frame = self.trackers[self.trackers[:,0]==frame]
            for det in dets_frame:
                x1,y1,x2,y2,vehicle_id = int(det[2]), int(det[3]), int(det[4]), int(det[5]), int(det[6])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                id_coord = (int((x1+x2)/2),int((y1+y2)/2))
                cv2.putText(img, str(vehicle_id), id_coord, cv2.FONT_HERSHEY_SIMPLEX, 1, color = (255, 0, 0))
                out.write(img)
        out.release()
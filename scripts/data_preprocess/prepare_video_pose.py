 # Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import torch
import numpy as np
import json
import copy
import torch
import random
import argparse
import shutil
import tempfile
import subprocess
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
import torch.multiprocessing as mp
import torch.distributed as dist
import pickle
import logging
from io import BytesIO
import os.path as osp
import multiprocessing as mp


import dwpose.util as util
from dwpose.wholebody import Wholebody


def get_logger(name="essmc2"):
    logger = logging.getLogger(name)
    logger.propagate = False
    if len(logger.handlers) == 0:
        std_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        std_handler.setFormatter(formatter)
        std_handler.setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
        logger.addHandler(std_handler)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Extract DWPose from videos")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/toy_train/svi-dance/raw/videos",
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="data/toy_train/svi-dance/preprocessed",
        help="Output directory for processed data"
    )
    return parser.parse_args()


class DWposeDetector:
    def __init__(self):

        self.pose_estimation = Wholebody()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            subset = subset[0][np.newaxis, :]
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18].copy()
            # print(score.shape)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            bodyfoot_score = subset[:,:24].copy()
            for i in range(len(bodyfoot_score)):
                for j in range(len(bodyfoot_score[i])):
                    if bodyfoot_score[i][j] > 0.3:
                        bodyfoot_score[i][j] = int(18*i+j)
                    else:
                        bodyfoot_score[i][j] = -1
            if -1 not in bodyfoot_score[:,18] and -1 not in bodyfoot_score[:,19]:
                bodyfoot_score[:,18] = np.array([18.]) # (bodyfoot_score[:,18]+bodyfoot_score[:,19])/2
            else:
                bodyfoot_score[:,18] = np.array([-1.])
            if -1 not in bodyfoot_score[:,21] and -1 not in bodyfoot_score[:,22]:
                bodyfoot_score[:,19] = np.array([19.]) # (bodyfoot_score[:,21]+bodyfoot_score[:,22])/2
            else:
                bodyfoot_score[:,19] = np.array([-1.])
            bodyfoot_score = bodyfoot_score[:, :20]

            bodyfoot = candidate[:,:24].copy()
            
            for i in range(nums):
                if -1 not in bodyfoot[i][18] and -1 not in bodyfoot[i][19]:
                    bodyfoot[i][18] = (bodyfoot[i][18]+bodyfoot[i][19])/2
                else:
                    bodyfoot[i][18] = np.array([-1., -1.])
                if -1 not in bodyfoot[i][21] and -1 not in bodyfoot[i][22]:
                    bodyfoot[i][19] = (bodyfoot[i][21]+bodyfoot[i][22])/2
                else:
                    bodyfoot[i][19] = np.array([-1., -1.])
            
            bodyfoot = bodyfoot[:,:20,:]
            bodyfoot = bodyfoot.reshape(nums*20, locs)

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            # bodies = dict(candidate=body, subset=score)
            bodies = dict(candidate=bodyfoot, subset=bodyfoot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            return draw_pose(pose, H, W)
            # return pose


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    # canvas = util.draw_bodypose(canvas, candidate, subset)
    canvas = util.draw_body_and_foot(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas_without_face = copy.deepcopy(canvas)

    canvas = util.draw_facepose(canvas, faces)

    return canvas_without_face, canvas

def dw_func(_id, file_path, dwpose_model, output_dir, dwpose_woface_folder='tmp_dwpose_wo_face', dwpose_withface_folder='tmp_dwpose_with_face'):
    video_name = (file_path).split('/')[-1].split('.mp4')[0]
    dwpose_woface_folder = os.path.join(output_dir, video_name)
    
    
    frame_all = []
    videoCapture = cv2.VideoCapture(file_path)
    iiii = 0
    while videoCapture.isOpened():
        # get a frame
        ret, frame = videoCapture.read()
        iiii += 1
        if ret:
            
            frame_all.append(frame)
        else:
            break
    
    videoCapture.release()

    
    results_vis = []
    video_frame_all = {}
    pose_frame_all = {}
    pose_frame_all_face = {}
    
    if len(frame_all)<2000:
        for i_index, frame in enumerate(frame_all):
            frame_name  = str(i_index).zfill(6)+".jpg"

            
            if frame.shape[1]>frame.shape[0]:
                margin = (frame.shape[1] - frame.shape[0])//2
                frame = frame[:,margin:-margin]

            frame_h, frame_w, _ = frame.shape
            if frame_w>=2048:
                frame = cv2.resize(frame,(frame_w//2,frame_h//2)) 
            

            _, img_encode = cv2.imencode('.jpg', frame)
            img_bytes = img_encode.tobytes()
            video_frame_all[frame_name] = img_bytes

            dwpose_woface, dwpose_wface = dwpose_model(frame)
            
            _, img_encode = cv2.imencode('.jpg', dwpose_woface)
            img_bytes = img_encode.tobytes()
            pose_frame_all[frame_name] = img_bytes

            _, img_encode = cv2.imencode('.jpg', dwpose_wface)
            img_bytes = img_encode.tobytes()
            pose_frame_all_face[frame_name] = img_bytes
            
            

        os.makedirs(dwpose_woface_folder, exist_ok=True)
        with open(os.path.join(dwpose_woface_folder+'/frame_data.pkl'), "wb") as tf:
            pickle.dump(video_frame_all,tf)

        with open(os.path.join(dwpose_woface_folder+'/dw_pose_with_foot_wo_face.pkl'), "wb") as tf:
            pickle.dump(pose_frame_all,tf)
        
        with open(os.path.join(dwpose_woface_folder+'/dw_pose_with_foot_with_face.pkl'), "wb") as tf:
            pickle.dump(pose_frame_all_face,tf)
        

def mp_main(dwpose_model, video_paths, posevideo_dir, output_dir):
    
    
    dwpose_model = DWposeDetector()  

    for i, file_path in enumerate(video_paths):
        
        file_path = os.path.join(posevideo_dir, file_path)
        
        logger.info(f"{i}/{len(video_paths)}, {file_path}")
        
        dw_func(i, file_path, dwpose_model, output_dir)



    
logger = get_logger('dw pose extraction')


if __name__=='__main__':
    # mp.set_start_method('spawn')
    
    args = parse_args()
    
    posevideo_dir = args.input_dir
    output_dir = args.output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    video_paths = os.listdir(posevideo_dir)
    video_list = [v for v in video_paths if v.endswith('.mp4')]
    
    logger.info("Input directory: {}".format(posevideo_dir))
    logger.info("Output directory: {}".format(output_dir))
    logger.info("There are {} videos for extracting poses".format(len(video_list)))

    logger.info('LOAD: DW Pose Model')
    
    dwpose_model = None

    mp_main(dwpose_model, video_list, posevideo_dir, output_dir)
    
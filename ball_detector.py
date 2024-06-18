from tracknet import BallTrackerNet
import torch
import cv2
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from model_lightning import LitTrackNetV2

class BallDetector:
    def __init__(self, path_model=None, device='cuda'):
        self.device = device
        if path_model:
            self.model = LitTrackNetV2.load_from_checkpoint(path_model, frame_in = 9, frame_out = 3)
            # self.model.load_state_dict(torch.load(path_model, map_location=device)['state_dict'], strict = True)
            # self.model = self.model.to(device)
            self.model.eval()
        self.width = 640
        self.height = 288

    def infer_model(self, frames, transform):
        """ Run pretrained model on a consecutive list of frames
        :params
            frames: list of consecutive video frames
        :return
            ball_track: list of detected ball points
        """
        ball_track = []
        scale = frames[0].shape[0] / self.height
        for num in tqdm(range(2, len(frames), 3)):
            img = frames[num]
            img_prev = frames[num-1]
            img_preprev = frames[num-2]
            image_list = [img_preprev, img_prev, img]
            imgs = torch.cat([transform(img)['image'] for img in image_list])
            inp = torch.unsqueeze(imgs, 0)

            out = self.model(inp.to(self.device))
            output = torch.sigmoid(out).detach().cpu().numpy()
            
            for i in range(3):
                x_pred, y_pred = self._detect_blob_concomp(output[0][i])
                ball_track.append((x_pred, y_pred))
        return ball_track

    def postprocess(self, feature_map, prev_pred, scale=1080/288, max_dist=80):
        """
        :params
            feature_map: feature map with shape (1,360,640)
            prev_pred: [x,y] coordinates of ball prediction from previous frame
            scale: scale for conversion to original shape (720,1280)
            max_dist: maximum distance from previous ball detection to remove outliers
        :return
            x,y ball coordinates
        """
        # feature_map *= 255
        # feature_map = feature_map.reshape((self.height, self.width))
        # feature_map = feature_map.astype(np.uint8)
        ret, heatmap = cv2.threshold(feature_map, 0.5, 1, cv2.THRESH_BINARY)
        heatmap = heatmap.astype(np.uint8)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                                   maxRadius=7)
        x, y = None, None
        if circles is not None:
            if prev_pred[0]:
                for i in range(len(circles[0])):
                    x_temp = circles[0][i][0]*scale
                    y_temp = circles[0][i][1]*scale
                    dist = distance.euclidean((x_temp, y_temp), prev_pred)
                    if dist < max_dist:
                        x, y = x_temp, y_temp
                        break                
            else:
                x = circles[0][0][0]*scale
                y = circles[0][0][1]*scale
        return x, y
    
    def _detect_blob_concomp(self, hm, _score_threshold = 0.5, _use_hm_weight = False, scale = 1080/288):
        x, y = None, None
        if np.max(hm) > _score_threshold:
            best_score = -1
            visi = True
            th, hm_th        = cv2.threshold(hm, _score_threshold, 1, cv2.THRESH_BINARY)
            n_labels, labels = cv2.connectedComponents(hm_th.astype(np.uint8))
            for m in range(1, n_labels):
                ys, xs = np.where(labels == m)
                ws     = hm[ys, xs]
                if _use_hm_weight:
                    score  = ws.sum()
                    x_temp      = np.sum( np.array(xs) * ws ) / np.sum(ws)
                    y_temp      = np.sum( np.array(ys) * ws ) / np.sum(ws)
                else:
                    score  = ws.shape[0]
                    x_temp      = np.sum( np.array(xs) ) / ws.shape[0]
                    y_temp      = np.sum( np.array(ys) ) / ws.shape[0]
                    #print(xs, ys)
                    #print(score, x, y)
                if score > best_score:
                    best_score = score
                    x, y = x_temp, y_temp
            return x * scale, y * scale
        return x, y
import cv2
import numpy as np
import torch
from model_lightning import LitTrackNetV2
import torch.nn.functional as F
from tracknet import BallTrackerNet
from tqdm import tqdm
from postprocess import refine_kps
from homography import get_trans_matrix, refer_kps

class CourtDetectorNet():
    def __init__(self, path_model=None,  device='cuda'):
        try:
            self.device = device
            if path_model:
                self.model = LitTrackNetV2.load_from_checkpoint(path_model, frame_in = 9, frame_out = 45)
                self.model.eval()
        except:
            self.model = BallTrackerNet(out_channels=15)
            self.device = device
            if path_model:
                self.model.load_state_dict(torch.load(path_model, map_location=device))
                self.model = self.model.to(device)
                self.model.eval()
            
    def infer_model(self, frames, transform):
        output_width = 640
        output_height = 360
        scale = frames[0].shape[0] / output_height
        
        kps_res = []
        matrixes_res = []
        if isinstance(self.model, LitTrackNetV2):
            for num in tqdm(range(2, len(frames), 3)):
                if num < 3:
                    img = frames[num]
                    img_prev = frames[num-1]
                    img_preprev = frames[num-2]
                    image_list = [img_preprev, img_prev, img]
                    imgs = torch.cat([transform(image = img)['image'] for img in image_list])
                    
                    inp = torch.unsqueeze(imgs, 0)

                    out = self.model(inp.to(self.device))
                    pred = F.sigmoid(out).detach().cpu().numpy()

                    
                    for i in range(3):
                        points = []
                        for kps_num in range(14):
                            x_pred, y_pred = self._detect_blob_concomp(pred[0][i * 15 + kps_num])
                            if x_pred is not None:
                                if kps_num not in [8, 12, 9]:
                                    x_pred, y_pred = refine_kps(image_list[i], int(y_pred), int(x_pred), crop_size=40)
                                points.append((x_pred, y_pred))                
                            else:
                                points.append(None)

                        matrix_trans = get_trans_matrix(points) 
                        points = None
                        if matrix_trans is not None:
                            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                            matrix_trans = cv2.invert(matrix_trans)[1]
                        kps_res.append(points)
                        matrixes_res.append(matrix_trans)
            kps_res = kps_res *  (num // 3)
            matrixes_res = matrixes_res * (num // 3)
            print(len(kps_res), len(matrixes_res))

        else:
            for num_frame, image in enumerate(tqdm(frames)):
                img = cv2.resize(image, (output_width, output_height))
                inp = (img.astype(np.float32) / 255.)
                inp = torch.tensor(np.rollaxis(inp, 2, 0))
                inp = inp.unsqueeze(0)

                out = self.model(inp.float().to(self.device))[0]
                pred = F.sigmoid(out).detach().cpu().numpy()

                points = []
                for kps_num in range(14):
                    heatmap = (pred[kps_num]*255).astype(np.uint8)
                    ret, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
                    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2,
                                            minRadius=10, maxRadius=25)
                    if circles is not None:
                        x_pred = circles[0][0][0]*scale
                        y_pred = circles[0][0][1]*scale
                        if kps_num not in [8, 12, 9]:
                            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred), crop_size=40)
                        points.append((x_pred, y_pred))                
                    else:
                        points.append(None)

                matrix_trans = get_trans_matrix(points) 
                points = None
                if matrix_trans is not None:
                    points = cv2.perspectiveTransform(refer_kps, matrix_trans)
                    matrix_trans = cv2.invert(matrix_trans)[1]
                kps_res.append(points)
                matrixes_res.append(matrix_trans)
        return matrixes_res, kps_res    

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
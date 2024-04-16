import cv2
import numpy as np
import torch
from model_lightning import LitTrackNetV2
import torch.nn.functional as F
from tqdm import tqdm
from postprocess import refine_kps
from homography import get_trans_matrix, refer_kps

class CourtDetectorNet():
    def __init__(self, path_model=None,  device='cuda'):
        self.device = device
        if path_model:
            self.model = LitTrackNetV2.load_from_checkpoint(path_model, frame_in = 9, frame_out = 45)
            self.model.eval()
            
    def infer_model(self, frames, images):
        output_width = 640
        output_height = 360
        scale = 3
        
        kps_res = []
        matrixes_res = []
        for num in tqdm(range(2, len(frames), 3)):
            img = frames[num]
            img_prev = frames[num-1]
            img_preprev = frames[num-2]
            imgs = torch.cat((img_preprev, img_prev, img))
            inp = torch.unsqueeze(imgs, 0)

            image = images[num]
            image_prev = images[num-1]
            image_preprev = images[num-2]
            image_list = [image_preprev, image_prev, image]

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
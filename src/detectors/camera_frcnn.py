"""Frozen camera detector: torchvision Faster R-CNN ResNet-50-FPN v2, COCO weights
(fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth, Li et al. 2021 "Benchmarking detection transfer
learning with ViT"), plus a *camera-only* monocular lift of each 2D box into the LiDAR frame.

The lift uses no LiDAR data at inference — only the static calibration, the ego poses, and a flat
road plane at a fixed height (estimated once from training-split labels). Two independent range
estimates are formed and both are kept:
  * ground-contact: intersect the ray through the box's bottom-centre pixel with the road plane,
  * size prior:     r = f * H_class / h_px.
Their disagreement is itself a camera-internal consistency signal used by the arbitration head.
"""
import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from ..geometry import apply_tf, inv_tf

# COCO category id -> (project class id, class-prior length, width, height)
COCO_MAP = {3: (0, 4.5, 1.9, 1.6),   # car
            8: (0, 5.2, 2.0, 1.9),   # truck -> Car (pickups / vans are labelled Car in Boreas)
            1: (1, 0.6, 0.6, 1.72),  # person
            2: (2, 1.8, 0.6, 1.7),   # bicycle
            4: (2, 2.0, 0.8, 1.6)}   # motorcycle


class CameraDetector:
    def __init__(self, device="cpu", score_thr=0.05, max_det=100, ground_z=-2.31, bottom_sigma_px=2.0):
        self.device = torch.device(device)
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_score_thresh=score_thr, box_detections_per_img=max_det)
        self.model.eval().to(self.device)
        self.ground_z = ground_z
        self.bottom_sigma_px = bottom_sigma_px

    @torch.no_grad()
    def detect_2d(self, img_rgb):
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        out = self.model([t])[0]
        return {k: v.cpu().numpy() for k, v in out.items()}

    def __call__(self, img_rgb, T_cam_lidar, P):
        raw = self.detect_2d(img_rgb)
        return self.lift(raw, T_cam_lidar, P, img_rgb.shape[1], img_rgb.shape[0])

    def lift(self, raw, T_cam_lidar, P, width, height):
        keep = np.array([int(c) in COCO_MAP for c in raw["labels"]], dtype=bool)
        boxes2d, scores, coco = raw["boxes"][keep], raw["scores"][keep], raw["labels"][keep]
        # COCO heads fire 'car' and 'truck' on the same vehicle; after mapping both are Car, so apply
        # class-aware NMS on the *project* classes.
        if len(boxes2d):
            proj = np.array([COCO_MAP[int(c)][0] for c in coco])
            k = torchvision.ops.batched_nms(torch.from_numpy(boxes2d).float(), torch.from_numpy(scores).float(),
                                            torch.from_numpy(proj), 0.7).numpy()
            boxes2d, scores, coco = boxes2d[k], scores[k], coco[k]
        n = len(boxes2d)
        out = {"boxes": np.zeros((n, 7)), "scores": scores.astype(np.float64), "labels": np.zeros(n, int),
               "boxes2d": boxes2d.astype(np.float64), "range_ground": np.full(n, np.nan),
               "range_size": np.full(n, np.nan), "range_sigma": np.full(n, np.nan),
               "azimuth": np.zeros(n), "truncated": np.zeros(n)}
        if n == 0:
            return out
        K_inv = np.linalg.inv(P[:3, :3])
        T_lidar_cam = inv_tf(T_cam_lidar)
        cam_origin = T_lidar_cam[:3, 3]
        fy = P[1, 1]
        for i in range(n):
            cid, L, W, H = COCO_MAP[int(coco[i])]
            u0, v0, u1, v1 = boxes2d[i]
            uc = 0.5 * (u0 + u1)
            ray_c = K_inv @ np.array([uc, v1, 1.0])
            ray_l = T_lidar_cam[:3, :3] @ ray_c
            # ground-contact range
            r_ground = np.nan
            if ray_l[2] < -1e-6:
                s = (self.ground_z - cam_origin[2]) / ray_l[2]
                if s > 0:
                    hit = cam_origin + s * ray_l
                    r_ground = float(np.hypot(hit[0], hit[1]))
            # size-prior range (depth along optical axis, then horizontal)
            h_px = max(v1 - v0, 1.0)
            depth = fy * H / h_px
            centre_ray_c = K_inv @ np.array([uc, 0.5 * (v0 + v1), 1.0])
            p_size = apply_tf(T_lidar_cam, (centre_ray_c * depth / centre_ray_c[2])[None])[0]
            r_size = float(np.hypot(p_size[0], p_size[1]))
            az_ray = T_lidar_cam[:3, :3] @ centre_ray_c
            azimuth = float(np.arctan2(az_ray[1], az_ray[0]))
            truncated = float(u0 <= 2 or v0 <= 2 or u1 >= width - 3 or v1 >= height - 3)
            # bottom-contact is unreliable when truncated at the image bottom
            use_ground = np.isfinite(r_ground) and v1 < height - 3 and r_ground < 150
            if use_ground:
                # range sigma from bottom-edge pixel noise: dr/dv ~ r^2 / (f * cam_height)
                cam_h = cam_origin[2] - self.ground_z
                sigma_g = (r_ground ** 2) * self.bottom_sigma_px / (fy * max(cam_h, 0.5))
                sigma_s = 0.15 * r_size
                w_g, w_s = 1 / max(sigma_g, 0.05) ** 2, 1 / max(sigma_s, 0.05) ** 2
                r = (w_g * r_ground + w_s * r_size) / (w_g + w_s)
                sigma = (w_g + w_s) ** -0.5
            else:
                r, sigma = r_size, 0.15 * r_size
            # the visible bottom edge is the near face: push the centre back by half the depth extent
            r_centre = r + 0.25 * (L + W)
            x, y = r_centre * np.cos(azimuth), r_centre * np.sin(azimuth)
            out["boxes"][i] = [x, y, self.ground_z + H / 2, L, W, H, azimuth]
            out["labels"][i] = cid
            out["range_ground"][i] = r_ground
            out["range_size"][i] = r_size
            out["range_sigma"][i] = sigma
            out["azimuth"][i] = azimuth
            out["truncated"][i] = truncated
        return out

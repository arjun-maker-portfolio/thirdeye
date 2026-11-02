import os

import cv2
import numpy as np

CLASS_NAMES = [
    "car_back",
    "car_side",
    "car_front",
    "bus_back",
    "bus_side",
    "bus_front",
    "truck_back",
    "truck_side",
    "truck_front",
    "motorcycle_back",
    "motorcycle_side",
    "motorcycle_front",
    "bicycle_back",
    "bicycle_side",
    "bicycle_front",
]


NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


CAR_CLASS_IDS = {NAME_TO_ID["car_back"], NAME_TO_ID["car_side"], NAME_TO_ID["car_front"]}


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (round(w0 * r), round(h0 * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def nms_np(boxes, scores, iou_thres=0.45, max_det=300):
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep


class Detector:
    def __init__(self, tflite_path, num_threads: int = 0):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow import lite

            Interpreter = lite.Interpreter
        nt = (os.cpu_count() or 4) if num_threads in (None, 0) else int(num_threads)
        try:
            self.interp = Interpreter(model_path=str(tflite_path), num_threads=nt)
        except TypeError:
            self.interp = Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()
        in0 = self.input_details[0]
        self.in_h = int(in0["shape"][1])
        self.in_w = int(in0["shape"][2])
        self.fp16 = in0["dtype"] == np.float16

    def predict(self, frame_bgr, conf_thres=0.25, iou_thres=0.45):
        img0 = frame_bgr
        lb, r, (dw, dh) = letterbox(img0, (self.in_h, self.in_w))
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        if self.fp16:
            x = x.astype(np.float16)

        self.interp.set_tensor(self.input_details[0]["index"], x)
        self.interp.invoke()

        out = self.interp.get_tensor(self.output_details[0]["index"])
        out = out[0] if out.ndim == 3 else out  # rows, outputs

        C = out.shape[1]
        xywh = out[:, 0:4]
        if C >= 6:
            obj = out[:, 4]
            if C > 6:
                cls_scores = out[:, 5:]
                cls_ids = np.argmax(cls_scores, axis=1)
                cls_confs = cls_scores[np.arange(out.shape[0]), cls_ids]
                scores = obj * cls_confs
            else:
                cls_ids = out[:, 5].astype(int)
                scores = obj
        else:
            scores = out[:, 4]
            cls_ids = np.zeros((out.shape[0],), dtype=int)

        # if normalized, scale to input pixels
        if np.max(xywh) <= 2.0:
            xywh[:, [0, 2]] *= self.in_w
            xywh[:, [1, 3]] *= self.in_h

        m = scores >= conf_thres
        xywh, scores, cls_ids = xywh[m], scores[m], cls_ids[m]
        if xywh.shape[0] == 0:
            return []

        x_c, y_c, w, h = xywh.T
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= r
        H, W = img0.shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, W - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, H - 1)

        keep = nms_np(boxes, scores, iou_thres=iou_thres)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

        return [
            (int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s), int(c))
            for b, s, c in zip(boxes, scores, cls_ids)
        ]


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

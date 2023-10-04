import torch
import numpy as np
import sys
sys.path.append("../")

from src.yolov5.models.common import DetectMultiBackend
from src.yolov5.utils.augmentations import letterbox
from src.yolov5.utils.plots import Annotator, colors
from src.yolov5.utils.torch_utils import select_device
from src.yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes


def load_model(device, data, weights, half, img_size):
    dnn = False
    device = select_device(device, 1)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    img_sz = [img_size, img_size]
    imgsz = check_img_size(img_sz, s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))
    return model, stride, names


def detect(model, im0, img_size, conf_thres, iou_thres, max_det, names):
    augment = False
    visualize = False
    classes = None
    agnostic_nms = False
    auto = True
    image = letterbox(im0, (img_size, img_size), stride=model.stride, auto=auto)[0]  # padded resize.
    image = image.transpose((2, 0, 1))  # CHW.
    image = np.ascontiguousarray(image)  # contiguous.
    image = torch.from_numpy(image).to(model.device)
    image = image.half() if model.fp16 else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0
    if len(image.shape) == 3:
        image = image[None]  # expand for batch dim
    pred = model(image, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    annotator = Annotator(np.ascontiguousarray(im0), line_width=2, example=str(names))
    people_list = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(image.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                if label == "person":
                    people_list.append(xyxy)
                annotator.box_label(xyxy, f'{label} {conf:.2f}', color=colors(int(cls), True))
    return annotator.result(), people_list

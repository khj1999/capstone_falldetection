from flask import Flask
from flask import render_template, send_file, Response

## 

import cv2
import time
import torch
import argparse
import numpy as np
from pickle import FALSE
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer

# =1.0=Import=Libraries======================
from trainer import Get_coord, draw_border
from PIL import ImageFont, ImageDraw, Image
# ===========================================


@torch.no_grad()
def gen_frames(camera):
    device = select_device('0')
    half = device.type != 'cpu'
    model = attempt_load('yolov7-w6-pose.pt', map_location=device)  # load FP32 model
    _ = model.eval()

    cap = cv2.VideoCapture(camera)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    
    while cap.isOpened:
        ret, frame = cap.read()
        if ret:
            orig_image = frame

            # preprocess frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))

            image = image.to(device)
            image = image.float()

            with torch.no_grad():
                output, _ = model(image)

            output = non_max_suppression_kpt(
                output, 0.5, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
            img = image[0].permute(1, 2, 0) * 255
            img = img.cpu().numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # =2.0=load fall icon ===================
            icon = cv2.imread("icon3.png")
            #========================================

            # =3.0= Fall detection ==================
            thre = (frame_height//2)+100
            for idx in range(output.shape[0]):
                kpts = output[idx, 7:].T
                plot_skeleton_kpts(img, kpts, 3)
                xmin, ymin = (output[idx, 2]-output[idx, 4] /
                              2), (output[idx, 3]-output[idx, 5]/2)
                xmax, ymax = (output[idx, 2]+output[idx, 4] /
                              2), (output[idx, 3]+output[idx, 5]/2)
                p1 = (int(xmin), int(ymin))
                p2 = (int(xmax), int(ymax))
                dx = int(xmax) - int(xmin)
                dy = int(ymax) - int(ymin)
                cx = int(xmin+xmax)//2
                cy = int(ymin+ymax)//2
                icon = cv2.resize(
                    icon, (50, 50), interpolation=cv2.INTER_LINEAR)
                difference = dy-dx
                ph = Get_coord(kpts, 2)
                if ((difference < 0) and (int(ph) > thre)) or (difference < 0):
                    # fall 감지
                    draw_border(img, p1, p2, (84, 61, 247), 10, 25, 25)
                    im = Image.fromarray(img)
                    draw = ImageDraw.Draw(im)
                    draw.rounded_rectangle((cx-10, cy-10, cx+60, cy+60), fill=(84, 61, 247),
                                           radius=15)
                    img = np.array(im)
                    img[cy:cy+50, cx:cx+50] = icon
                    cv2.putText(img, "fall detection",(30, 130), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                # =========================================
            img_ = img.copy()
            img_ = cv2.resize(img_, (960, 540), interpolation=cv2.INTER_LINEAR)
            
            out_ret, buffer = cv2.imencode('.jpg', img_)
            out_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')
                
        else:
            break

app = Flask(__name__)

@app.route("/")
def video():
    return render_template("video.html")

@app.route("/video_feed1")
def video_feed1():
    return Response(gen_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed2")
def video_feed2():
    return Response(gen_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
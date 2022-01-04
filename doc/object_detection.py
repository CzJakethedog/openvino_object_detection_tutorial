"""General library 
"""
import sys, os
import logging
from time import time
from pathlib import Path
#sys.path.append(str(Path(__file__).resolve().parents[0] / 'utils'))
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

"""
OpenVINO, OpenCV, numpy library
"""
from openvino.inference_engine import IECore
import cv2
import numpy as np 

"""
 Parameter list
"""
model_path = r""
weight_path = r""
input_src = 0
#input_src = r""
label_mapping_path = r""
device = 'CPU'
prob_threshold = 0.5
max_num_request = 2

def print__version():
    log.info(f"Python : {sys.version}")
    log.info(f"OpenCV : {cv2.__version__}")
    log.info(f"Current path : {Path(__file__).resolve().parents[0]}")

class SSD():
    def __init__(self, ie, model_path, weight_path):
        self.net = ie.read_network(model = model_path, weights = weight_path)
        self.input_blob_name = next(iter(self.net.input_info)) #get input name from the model, iter() - returns an iterator for the given object
        self.output_blob_name = next(iter(self.net.outputs)) #get output name from the model 
        self.n, self.c, self.h, self.w = self.net.input_info[self.input_blob_name].input_data.shape
   
    def model_details(self):
        return log.info(f"Height: {self.h}, Width: {self.w}, Channel: {self.c}, Batch Size: {self.n}")
"""
Utils function
"""
def get_model(ie, model_path, weight_path):
    return SSD(ie, model_path, weight_path)
    
def load_label(label_mapping_path):
    if os.path.isfile(label_mapping_path):
        with open(label_mapping_path, 'r') as f:
            return [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        print("Loaded label mapping file [",label_mapping_path,"]")
    else:
        print("No label mapping file has been loaded, only numbers will be used",
            " for detected object labels")

"""
Image Processing function
"""
def preprocess(inputs, size, shape):
    #shape in cv2 is height, weight, channel
    resized_image = cv2.resize(inputs,size)
    # h, w = resized_image.shape[:2]
    resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    resized_image = resized_image.reshape(shape)
    return resized_image

def open_image_capture(input_src):
    if os.path.isfile(input_src):
        image=True
        return cv2.imread(input_src, cv2.IMREAD_COLOR),image #(num_rows, num_cols, num_channels)
    elif input_src == 0:
        image=False
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        status = cap.open(input_src)
        if not status:
            raise log.info(f"Can't find the camera {input_src}")
        return cap, image
    else:
        raise log.info(f"Input {input_src} is invalid.")
 
def plot_bbox(frame, bbox, label, prob):
    font=cv2.FONT_HERSHEY_DUPLEX
    font_size=0.8
    font_thickness=1
    text_color=(0, 0, 255) #cv2: BGR
    box_color = (0, 255, 0) 
    text_label = '{}: {:.1f}'.format(label, round(prob * 100, 1))
    x_min, y_min, x_max, y_max = bbox
    text_size = cv2.getTextSize(text_label, fontFace=font, fontScale=font_size, thickness=font_thickness)
    frame = cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(x_max, y_max), color=box_color, thickness=2)
    frame = cv2.rectangle(frame, pt1=(x_min, y_min), pt2=(x_min+text_size[0][0], y_min-text_size[0][1] - 7), color=box_color, thickness=cv2.FILLED)
    frame = cv2.putText(frame, text=text_label, org=(x_min, y_min - 7), fontFace=font, fontScale=font_size, color=text_color, thickness=1) 
    
    return frame

def postprocess(frame, detections, labels, fps, input_width, input_height):
    """
    Post process for drawing bounding box and text.
    """
    font=cv2.FONT_HERSHEY_DUPLEX
    font_size=0.6
    font_thickness=1
    text_color=(255, 0, 0) #cv2: BGR
    text_fps =  'FPS: {:.1f}'.format(fps)
    text_org = (int(input_width*0.05), int(input_height*0.05))
    for detection in detections:
        prob = float(detection[2])
        if prob > prob_threshold:
            objLabel = labels[int(detection[1])]
            #bbox = (x_min, y_min, x_max, y_max)
            bbox = (int(detection[3] * input_width), 
                    int(detection[4] * input_height),
                    int(detection[5] * input_width),
                    int(detection[6] * input_height))
            # print(objLabel) show the result 
            frame = plot_bbox(frame, bbox, label=objLabel, prob=prob) 
    frame_to_show = cv2.putText(frame, text=text_fps, org=text_org, fontFace=font, fontScale=font_size, color=text_color, thickness=1) 
    return frame_to_show
"""
Main function
"""

def main():
    log.info('Initializing Inference Engine...')
    ie = IECore()
    log.info('Reading network from IR...')
    model = get_model(ie=ie, model_path=model_path, weight_path=weight_path)
    labels = load_label(label_mapping_path=label_mapping_path)
    
    exec_net = ie.load_network(network=model.net, device_name=device, num_requests=max_num_request)

    cap, image = open_image_capture(input_src=input_src)
    if not image:
        input_width = cap.get(3)
        input_height = cap.get(4)    
    else:
        print(cap.shape)
        input_width = cap.shape[1]
        input_height = cap.shape[0]  
    log.info('Starting inference...')
    print("To close the application, switch to the output window and press ESC key or 'q' key")
   
    while True:
 
        if image == False:
            _,frame = cap.read() #event, frame: we only need frame here
        else:
            frame = cap
        if frame is None:
            raise ValueError("Can't read an image from the input")
            break
            
        in_frame = preprocess(frame, (model.w, model.h), (model.n, model.c, model.h, model.w))
        start_time = time()
        """
        model.input_blob_name  : input name in dict of model
        model.output_blob_name : output name in dict of model
        """
        exec_net.requests[0].infer(inputs={model.input_blob_name: in_frame})
        res = exec_net.requests[0].outputs[model.output_blob_name]
        obj = np.squeeze(res) #np.squeeze to remove redundant array dimension 
        fps = 1/(time()-start_time)
        frame = postprocess(frame=frame, detections=obj, labels=labels, fps=fps, input_width=input_width, input_height=input_height)
        
        cv2.imshow('Detection Results', frame)
        ESC_KEY = 27
        if not image:
            key = cv2.waitKey(1) & 0xFF
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                cap.release()
                break
        else:
            key = cv2.waitKey(0)
            if key in {ord('q'), ord('Q'), ESC_KEY}: break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main() or 0)
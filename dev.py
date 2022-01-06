"""
General library 
"""
import sys, os
import logging
from time import time
from pathlib import Path
logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

"""
OpenVINO, OpenCV, numpy library
"""


"""
 Parameter list
"""
# model_path = r""
# weight_path = r""
# input_src = 
# label_mapping_path = r""
# device = 
# prob_threshold = 
# max_num_request = 

def print__version():
    """
    log all the informations
    """
    pass

class SSD():
    def __init__(self, ie, model_path, weight_path):
    
        """
        Args : 
            ie : OpenVINO inference engine object
            model_path : str, path to the model xml
            weight_path : str, path to the model bin
        
        - initialize the ie network 
        - get the input blob name, output blob name
        - get the batch, channel, height, weight value from the model
        """
        self.net = #read network
        self.input_blob_name = #get input_info e from the model, iter() - returns an iterator for the given object
        self.output_blob_name = #get outputs from the model 
        self.n, self.c, self.h, self.w = # get input_data.shape from the input_info
   
    def model_details(self):
        return log.info(f"Height: {self.h}, Width: {self.w}, Channel: {self.c}, Batch Size: {self.n}")
"""
Utils function
"""
def get_model(ie, model_path, weight_path):
    """
    Args : 
        ie : OpenVINO inference engine object
        model_path : str, path to the model xml
        weight_path : str, path to the model bin
        
    Return:
        SSD() : Class method to object    
    """
    pass
    
def load_label(label_mapping_path):
    """
    Args:
        label_mapping_path : str, path to label mapping file.
    Return:
        list of string and index.
    """
    # load the label mapping path
    pass

"""
Image Processing function
"""
def preprocess(input_image, size, shape):
    """
    Args:
        input_image : input image from cv2 return
        size : size of model
        shape : shape of model, e.g: [n, c, h, w]
    Return:
        Output of resized image. 
    """
    #shape in cv2 is height, weight, channel

    # h, w = resized_image.shape[:2]
    # Change data layout from HWC to CHW, e.g: cv2.cv.transpose( src[, dst] )

    # Reshape the dimension of resized image to be same as model shape. 

    pass
def open_image_capture(input_src):
    """
    Args:
        input_src : "0" is using camera or path to the image (str)
    Return:
        cv2.imread output object or camera input
        image : Bool, to indicate using image or camera. 
    """
    # if-loop to check the input_src type either camera or image_path.
    image = True
    pass
 
def plot_bbox(frame, bbox, label, prob):
    """
    Args:
        frame : input src from cv2 object
        bbbox : bounding box value
        label : str, output from label mapping dict
        prob  : int, probability 
    Return:
        frame after post processing. 
    """
    # Dimension value for object detection display
    font=cv2.FONT_HERSHEY_DUPLEX
    font_size=0.8
    font_thickness=1
    text_color=(0, 0, 255) #cv2: BGR
    box_color = (0, 255, 0) 
    text_label = '{}: {:.1f}'.format(label, round(prob * 100, 1))
    
    # get the bbox dimension 
    # get the text size
    # draw rectangle to crop the detected object
    # draw rectangle and put text to show the object name 
    
    pass

def postprocess(frame, detections, labels, fps, input_width, input_height):
    """
    Wrapper:
        Post process for drawing bounding box and text.
        
    Agrs:
        frame        : input src from cv2 object
        detection    : list of result
        label        : str, output from label mapping dict
        fps          : int, frame per sec
        input_width  : int or float, original width value
        input_height : int or float, original height value
    Return:
        frame_to_show : output from post processed cv2 image.
    """
    # Dimension value for FPS display
    font=cv2.FONT_HERSHEY_DUPLEX
    font_size=0.6
    font_thickness=1
    text_color=(255, 0, 0) #cv2: BGR
    text_fps =  'FPS: {:.1f}'.format(fps)
    text_org = (int(input_width*0.05), int(input_height*0.05))
    
    # for loop the detections
    # detections[2] : check if prob is greater than prob_threshold
    # detections[1] : get the labels 
    # generate bbox dimension, e.g: detections[3-6], min_x, min_y, max_x, max_y
    # put fps text to the frame. 
    
    pass
"""
Main function
"""

def main():
    log.info("Initializing Inference Engine...")
    
    log.info("Loading the model...")
    
    log.info("Loading the label mapping information...")
    
    log.info("Reading network from IR...")
    
    log.info("Collecting the data from the cv2 input...")

    log.info("Retrieving the width and height data...")
    
    log.info('Starting inference...')
    print("To close the application, switch to the output window and press ESC key or 'q' key")
   
    while True:
        # check if input_src is image to get the capture object
        
        # close loop if error.
        if frame is None:
            # if next_frame_id == 0:
            raise ValueError("Can't read an image from the input")
            break
         
        # preprocess.
        
        # start counting the time for inferencing fps value.
        
        """
        model.input_blob_name  : input name in dict of model
        model.output_blob_name : output name in dict of model
        """
        # start inferencing.
        
        # get the result from executable network.
        
        # np.squeeze to remove redundant array dimension.
        
        # stop counting the time for inferencing fps value and calculate fps, fps = 1/processing_time.
        
        # post-processing to decorate the frame.
        
        # cv2.imshow to display the output result. e.g: cv2.imshow(str[window name], frame)
        
        # set a key way to exit the loop
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
import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet
# from darknet_images import image_detection

def parser1():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--type_detect", default="4",
                        help="Choice detector v4 or v3 default: yolov4")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--data_file", default="obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    return parser.parse_args()

args = parser1()


if args.type_detect == '4':
    config_file = './cfg/yolov4.cfg'
    weights = './model/yolov4.weights'
else:   
    config_file = './cfg/yolov3.cfg'
    weights = './model/yolov3.weights'
random.seed(3)
network, class_names, class_colors = darknet.load_network(
    config_file,
    args.data_file,
    weights,
    batch_size=args.batch_size
)

def image_detection(image, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    # if image_path.split('.')[-1] in ['jpg', 'jpeg', 'png']:
    # image = cv2.imread(image_path)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def pred(img):
    prev_time = time.time()
    image, detections = image_detection(
        img, network, class_names, class_colors, args.thresh
        )

    # if args.save_labels:
    #     save_annotations(image_name, image, detections, class_names)
    # darknet.print_detections(detections, args.ext_output)
    fps = int(1/(time.time() - prev_time))
    # print("FPS: {}".format(fps))
    return image, detections



if __name__ == "__main__":
    # main()
    img = cv2.imread('../train_data/day/test/cam_09_206.jpg')
    pred(img)



#python3 darknet_images.py --input ../train_data/day/test/cam_09_206.jpg --weights ../model/model_daynight_v3/yolov3_final.weights --data_file obj.data
#python3 main.py --input ../train_data/day/test/cam_09_206.jpg --weights ../model/model_daynight_v3/yolov3_final.weights --data_file obj.data

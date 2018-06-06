import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from YOLOv3 import Darknet
import pickle as pkl
import pandas as pd
import random
from matplotlib import pyplot as plt
import matplotlib
import copy


def arg_parse():
    """
    Parse arguments to the detect module
    
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.2)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.5)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


# reso default 416


args = arg_parse()
# images = args.images

# images = 'DOG.png'
# images = 'traffic_light.jpg'
# images = 'park.JPG'
# images = 'beach.JPG'
# images = 'desk.JPG'
images = 'Images/street6.png'
# images = 'office4.png'


batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

# Set up the neural network
print("Loading network.....")
# Set up the netowrk with the architecture specified for YOLOv3
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()

read_dir = time.time()

# Detection phase
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

# If loaction for detections not specified make it
if not os.path.exists(args.det):
    os.makedirs(args.det)

# Use OpenCV to load in the data
load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

# Pytorch variables for images
# Maps the image dimensions to 416 x416
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# List contaning dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

# Create the batches
leftover = 0
if len(im_dim_list) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                            len(im_batches))])) for i in range(num_batches)]

write = 0

# THIS IS THE DETECTION LOOP

# We iterate over the batches, generate the prediction, and 
# concatenate the prediction tensors (of shape, D x 8, the output of write_results 
# function) of all the images we have to perform detections upon.

if CUDA:
    im_dim_list = im_dim_list.cuda()

start_det_loop = time.time()
# Im_batches is a list of tensor images
# For a single image, im_batches is 1
for i, batch in enumerate(im_batches):
    # batch contains the images
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction, stored_layers = model(Variable(batch), CUDA)

    # prediction contains the data for each detection  - ALL THE NMS AND THRESHOLD IS DONE HERE
    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

    if not write:  # If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()
try:
    output
except NameError:
    print("No detections were made")
    exit()

# Rescaling the prediction boxes to the original image sizes

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

# Undo padding and rescaling
output[:, 1:5] /= scaling_factor

# Clip any bounding boxes that may have boundaries outside the image

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()

###############################################################################

# Plotting the activation maps
plt.close('all')
plt.figure(), plt.set_cmap('hot'), plt.imshow(stored_layers[10].numpy())

# Each detections has 8 attributes, namely, index of the image in the batch to which
# the detection belongs to, 4 corner coordinates, objectness score, the score of class
# with maximum confidence, and the index of that class.


# This function draws the bounding boxes

global_semantic_data = []




def write(x, results, input_img):
    # x is a rescaled bounding box for a single object in an image

    global global_semantic_data

    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    # Extracting index of the image in the batch
    img = results[int(x[0])]
    input_img = input_img[int(x[0])]
    # Extracting the index pertaining to the class of the object
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)

    bounding_box = input_img[c1[1]:c2[1], c1[0]:c2[0], :]

    global_semantic_data.append(bounding_box)
    cv2.imwrite('object.png', bounding_box)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);

    return img




# output contains the rescaled boudning boxes; loaded_ims are the input images
raw_img = copy.deepcopy(loaded_ims)
list(map(lambda x: write(x, loaded_ims, raw_img), output))

# Each image is saved by prefixing the 'det' in fornt of the image name

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

# Finally, write the images with detections to the address in det_names.

list(map(cv2.imwrite, det_names, loaded_ims))

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()

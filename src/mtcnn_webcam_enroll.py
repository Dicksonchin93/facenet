"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import align.extract_image_chips
import random
from time import sleep
import cv2
import dlib
from facealigner import FaceAligner

fileDir = os.path.dirname(os.path.realpath(__file__))
dlibModelDir = os.path.join(fileDir, 'models','dlib')


def main(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"  
    print('Creating networks and loading parameters')   
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_class_dir = os.path.join(output_dir, args.name)
    if not os.path.exists(output_class_dir):                                
        os.makedirs(output_class_dir)  
    filename = 0 
    predictor = dlib.shape_predictor(args.dlibFacePredictor)
    pose = FaceAligner(predictor)
    while True:
        ret, frame = video_capture.read()
        bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        #warped = (align.extract_image_chips.extract_image_chips(frame,np.transpose(points), args.image_size, 0.37)) 
            #bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            p = points[:,0:10]
            img_size = np.asarray(frame.shape)[0:2]
            for idx, bbs in enumerate(det):
                output_filename = os.path.join(output_class_dir, str(filename) + '.jpg')
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(bbs[0]-args.margin/2, 0)
                bb[1] = np.maximum(bbs[1]-args.margin/2, 0)
                bb[2] = np.minimum(bbs[2]+args.margin/2, img_size[1])
                bb[3] = np.minimum(bbs[3]+args.margin/2, img_size[0])
                #cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]
                #aligned = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                #prewhitened = facenet.prewhiten(aligned)
                #img_list[idx] = prewhitened  
                #cv2.imwrite(output_filename, aligned)
                rect = dlib.rectangle(left=long(bb[0]), top=long(bb[1]), right=long(bb[2]), bottom=long(bb[3]))
                shape, lm, p1, p2, euler_angles = pose.align(frame,frame, rect)
                cv2.imshow('warped', shape)             
                for i in range(5):
                    for j in range(len(p[i])):
                        x = int(p[i][j])
                        y = int(p[i+5][j])
                        #cv2.circle(frame, (x,y), 1, (0,0,255), 2)
                cv2.rectangle(frame,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(255,0,0),1)
                print(euler_angles)        
                for (x, y) in lm:
                    cv2.circle(frame, (x, y), 1,(0,0,255), -1)
                cv2.line(frame, p1, p2, (255,0,0), 2)
                filename += 1
  
           
        else:
            print('Unable to align')     
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--name', type=str, help='Person name')
    parser.add_argument(
       '--captureDevice',
       type=int,
       default=0,
       help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=20)
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
            
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


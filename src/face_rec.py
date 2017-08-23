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
import random
from time import sleep
import math
import cv2
import time
import dlib
from facealigner import FaceAligner


fileDir = os.path.dirname(os.path.realpath(__file__))
dlibModelDir = os.path.join(fileDir, 'models','dlib')

def main(args):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"  
    count = []
    fc = 0
    fps = 0
    start = time.time() 
    emb_array, labels, class_names = create_rep(args) 
    print("Time taken to create rep {}".format(time.time() - start)) 
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
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (-200.0, 170.0, -135.0),     # Left eye center
                            (200.0, 170.0, -135.0),      # Right eye center
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                         
                            ])
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(args.model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            predictor = dlib.shape_predictor(args.dlibFacePredictor)  # dlib initialize
            pose = FaceAligner(predictor)
            
            while True:
                loop_start = time.time()                    
                ret, frame = video_capture.read()
                size = frame.shape
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array(
                                         [[focal_length, 0, center[0]],
                                         [0, focal_length, center[1]],
                                         [0, 0, 1]], dtype = "double"
                                        )
        
                if np.sum(count) > 1.0:
                    fps = fc
                    fc = 0                  
                    count = [] 
                s = time.time()
                bounding_boxes, points = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces>0:
                    with tf.device('/cpu:0'):
                        det = bounding_boxes[:,0:4]
                        p = points[:,0:10]
                        img_size = np.asarray(frame.shape)[0:2]
                        img_list = [None] * nrof_faces
                        pts = []
                        for idx, bbs in enumerate(det):
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(bbs[0]-args.margin/2, 0)
                            bb[1] = np.maximum(bbs[1]-args.margin/2, 0)
                            bb[2] = np.minimum(bbs[2]+args.margin/2, img_size[1])
                            bb[3] = np.minimum(bbs[3]+args.margin/2, img_size[0])
                            cropped = frame[bb[1]:bb[3],bb[0]:bb[2],:]
                            aligned = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            rect = dlib.rectangle(left=long(bb[0]), top=long(bb[1]), right=long(bb[2]), bottom=long(bb[3]))
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            shape, lm, p1, p2, euler_angles, TOTAL, ear, leftEyePts, rightEyePts= pose.align(frame,gray, rect)
                            prewhitened = facenet.prewhiten(aligned)
                            img_list[idx] = prewhitened
                            cv2.rectangle(frame,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),(255,0,0),1)
                            #cv2.putText(frame, "p"+ str('%.3f' % euler_angles[0]) + "y" + str('%.3f' % euler_angles[1]) + "r" + str('%.3f' % euler_angles[2]) , (bb[0],bb[1]-60),
                            #            cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            #            color=(102, 255, 204), thickness=1)  
                            cv2.putText(frame, "Blinks ="+ str(TOTAL) , (150,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                        color=(102, 255, 204), thickness=1)  
                   
                            #for (x, y) in lm:
                            #    cv2.circle(frame, (x, y), 1,(0,0,255), -1)
                            for (x, y) in leftEyePts:
                                cv2.circle(frame, (x, y), 1,(0,0,255), -1)
                            for (x, y) in rightEyePts:
                                cv2.circle(frame, (x, y), 1,(0,0,255), -1)
                            image_points = np.array([
		                                             (p[2][idx], p[7][idx]),# Nose tip
		                                             (p[0][idx], p[5][idx]),# Left eye left center
		                                             (p[1][idx], p[6][idx]),# Right eye right center
		                                             (p[3][idx], p[8][idx]),# Left Mouth corner
		                                             (p[4][idx],p[9][idx]),# Right mouth corner
                                                    ], dtype="double") 
                            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                            R, _ = cv2.Rodrigues(rotation_vector)
                            sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
                            singular = sy < 1e-6   
                            if not singular:
		                        x = math.atan2(R[2,1] , R[2,2])
		                        y = math.atan2(-R[2,0], sy) 
		                        z = math.atan2(R[1,0], R[0,0]) 
                            else:
		                        x = math.atan2(-R[1,2], R[1,1])
		                        y = math.atan2(-R[2,0], sy)
		                        z = 0		            
                            #print (x, y, z)
                            cv2.putText(frame, "p"+ str('%.1f' % x) + "y" + str('%.1f' % y) + "r" + str('%.1f' % z),
                                        (bb[0],bb[1]-60),
                                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                                        color=(102, 105, 204), thickness=1) 
                            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                            p3 = ( int(image_points[0][0]), int(image_points[0][1]))
                            p4 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))                           
                            cv2.line(frame, p3, p4, (255,100,0), 2)
                            for i in range(5):
                                for j in range(len(p[i])):
                                    x = int(p[i][j])
                                    y = int(p[i+5][j])
                                    cv2.circle(frame, (x,y), 1, (0,0,255), 2)
                            
                            #cv2.line(frame, p1, p2, (255,0,0), 2)
                            pts.append([bb[0],bb[1],idx])
                            #cv2.imshow('warped', shape)
                    images = np.stack(img_list)
                   # print("time for detection and landmark localization {}".format(time.time() - s))
                    d = time.time()
                    with tf.device('/gpu:0'):
                        feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        counter = []
                        for reps in pts:
                            distance = []                       
                            for j in range(len(labels)):
                                dist = np.sqrt(np.sum(np.square(np.subtract(emb_array[j,:], emb[reps[2],:]))))
                                distance.append(dist)
                            conf = min(distance)
                            conf_index= distance.index(conf)
                            bbx = reps[0]                   
                            bby = reps[1]
                            print ("Distance to {} is {}".format(class_names[labels[conf_index]],conf))
                            if conf < args.threshold:
                                #counter.append([class_names[labels[conf_index]], conf])
                                cv2.putText(frame, class_names[labels[conf_index]], (bbx, bby),
                                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                                            color=(152, 255, 204), thickness=1)
                                cv2.putText(frame, str(conf), (bbx, bby-30),
                                            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                                            color=(152, 255, 204), thickness=1)
                    #print("time for recognition {}".format(time.time() - d))      
                count.append(time.time() - loop_start)
                fc += 1
                cv2.putText(frame, str(fps) + "fps" , (30,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                        color=(102, 255, 204), thickness=1)
            
            
            
                cv2.imshow('video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    video_capture.release()
    cv2.destroyAllWindows()
            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument(
       '--captureDevice',
       type=int,
       default=1,
       help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=20)
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned faces')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--threshold', type=float,
        help='Threshold for matching', default=1.0)
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    return parser.parse_args(argv)

def create_rep(args): 
    NUM_THREADS = 8
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, intra_op_parallelism_threads= NUM_THREADS, inter_op_parallelism_threads = NUM_THREADS)) as sess:
           
            dataset = facenet.get_dataset(args.data_dir)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print("Number of classes: {}".format(len(dataset)))
            print("Number of images: {}".format(len(paths)))
                      
            print('Loading feature extraction model')
            facenet.load_model(args.model)
                        
       
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            print(embedding_size)
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                print(images.shape)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                  
            
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]
    return emb_array, labels, class_names
         
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


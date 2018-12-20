"""
Pulse extraction using POS algorithm (%(version)s)
"""


import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("MacOSX")
from matplotlib import pyplot as plt

import os
import sys
sys.path.insert(0, './SkinDetector')
import pkg_resources

import numpy as np
import cv2
import dlib

from imutils.video import VideoStream
from imutils import face_utils
import imutils

import argparse
import skin_detector

def main(user_input=None):
    # EXTRACT PULSE
    pulsedir ="/Volumes/MacMini-Backups/siw-db/live/pulse/"
    start = 0
    end = 450
 
    framerate = 30

    # FREQUENCY ANALYSIS
    nsegments = 12
    
    plot =  False
    image_show = True

    left_increase_ratio = 0.05 #5%
    top_increase_ratio = 0.25 #5%
  
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help = "path to the (optional) video file")
    args = vars(ap.parse_args())


    if not args.get("video", False):
        from_webcam = True
        camera = cv2.VideoCapture(0)
        start = 0
        end = 450
	# otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])

    video_file_path = args["video"]
    video_file_name = os.path.basename(video_file_path)
    
    start_index = start
    end_index = end

    # number of final frames
    if end_index > 0:
        nb_frames = end_index - start_index


    # loop on video frames
    frame_counter = 0
    i = start_index

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
   
    while (i >= start_index and i < end_index):
        (grabbed, frame) = camera.read()
    
        if not grabbed:
            continue

        print("Processing frame %d/%d...", i+1, end_index)
        
        h,w,_ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects)==0:
            continue

        if image_show:
            show_frame = frame.copy()
       
        if(len(rects)>0):
            rect = rects[0] 
            '''          
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for counter,(x, y) in enumerate(shape):
                cv2.circle(show_frame, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(show_frame,str(counter),(x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1)
            '''
               
            
            left, right, top, bottom = rect.left(), rect.right(), rect.top(),rect.bottom()
            width = abs(right - left)
            height = abs(bottom - top)
            print("Left, right, top, bottom: ",left, right, top, bottom)
            #print("Width and Height of bounding box : ",width,height)
            
            face_left = int(left - (left_increase_ratio/2)*width)
            face_top = int(top - (top_increase_ratio)*height)
            #face_right = int(right + (area_increase_ratio/2)*width)
            #face_bottom = int(bottom + (area_increase_ratio/2)*height)
            
            face_right = right
            face_bottom = bottom
            
            print("Increased coordinates: ",face_left, face_right, face_top, face_bottom)
            
            if image_show:
                cv2.rectangle(show_frame,(left,top),(right,bottom),(255,255,0),3)
                cv2.rectangle(show_frame,(face_left,face_top),(face_right,face_bottom),(0,255,0),3)
            
            face = frame[face_top:face_bottom,face_left:face_right]
            
            if(face.size==0):
                continue
            #    continue
            #Extract face skin pixels
            mask = skin_detector.process(face)
     
            
            #print("Mask shape: ",mask.shape)
            masked_face = cv2.bitwise_and(face, face, mask=mask)
            number_of_skin_pixels = np.sum(mask>0)



            #compute mean
            r = np.sum(masked_face[:,:,2])/number_of_skin_pixels
            g = np.sum(masked_face[:,:,1])/number_of_skin_pixels 
            b = np.sum(masked_face[:,:,0])/number_of_skin_pixels

            if frame_counter==0:
                mean_rgb = np.array([r,g,b])
            else:
                mean_rgb = np.vstack((mean_rgb,np.array([r,g,b])))

            
            print("Mean RGB -> R = {0}, G = {1}, B = {2} ".format(r,g,b))

        if image_show:
            if h>w and h>640:
                    dim = (int(640 * (w/h)),640)    
                    show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)
            if w>h and w>640:
                    dim = (640, int(640 * (h/w)))
                    show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)
         
        #cv2.imshow("frame",show_frame)
        if(image_show):
            cv2.imshow("Masked face",masked_face)
            cv2.waitKey(1)
        frame_counter +=1
        i += 1
        #end loop
    
    camera.release()
    cv2.destroyAllWindows()

    if plot:
        f = np.arange(0,mean_rgb.shape[0])
        plt.plot(f, mean_rgb[:,0] , 'r', f,  mean_rgb[:,1], 'g', f,  mean_rgb[:,2], 'b')
        plt.title("Mean RGB - Complete")
        plt.show()

    #Calculating l
    l = int(framerate * 1.6)
    print("Window Length : ",l)

    H = np.zeros(mean_rgb.shape[0])

    for t in range(0, (mean_rgb.shape[0]-l)):
        #t = 0
        # Step 1: Spatial averaging
        C = mean_rgb[t:t+l-1,:].T
        #C = mean_rgb.T
        print("C shape", C.shape)
        print("t={0},t+l={1}".format(t,t+l))
        if t == 3:
            plot = False

        if plot:
            f = np.arange(0,C.shape[1])
            plt.plot(f, C[0,:] , 'r', f,  C[1,:], 'g', f,  C[2,:], 'b')
            plt.title("Mean RGB - Sliding Window")
            plt.show()
        
        #Step 2 : Temporal normalization
        mean_color = np.mean(C, axis=1)
        #print("Mean color", mean_color)
        
        diag_mean_color = np.diag(mean_color)
        #print("Diagonal",diag_mean_color)
        
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        #print("Inverse",diag_mean_color_inv)
        
        Cn = np.matmul(diag_mean_color_inv,C)
        #Cn = diag_mean_color_inv@C
        #print("Temporal normalization", Cn)
        #print("Cn shape", Cn.shape)

        if plot:
            f = np.arange(0,Cn.shape[1])
            #plt.ylim(0,100000)
            plt.plot(f, Cn[0,:] , 'r', f,  Cn[1,:], 'g', f,  Cn[2,:], 'b')
            plt.title("Temporal normalization - Sliding Window")
            plt.show()
    
        #Step 3: 
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix,Cn)
        #S = projection_matrix@Cn
        print("S matrix",S)
        print("S shape", S.shape)
        if plot:
            f = np.arange(0,S.shape[1])
            #plt.ylim(0,100000)
            plt.plot(f, S[0,:] , 'c', f,  S[1,:], 'm')
            plt.title("Projection matrix")
            plt.show()

        #Step 4:
        #2D signal to 1D signal
        std = np.array([1,np.std(S[0,:])/np.std(S[1,:])])
        print("std",std)
        P = np.matmul(std,S)
        #P = std@S
        print("P",P)
        if plot:
            f = np.arange(0,len(P))
            plt.plot(f, P, 'k')
            plt.title("Alpha tuning")
            plt.show()

        #Step 5: Overlap-Adding
        H[t:t+l-1] = H[t:t+l-1] +  (P-np.mean(P))/np.std(P)

    print("Pulse",H)
    signal = H
    print("Pulse shape", H.shape)
 

    #FFT to find the maxiumum frequency
    # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
    segment_length = (2*signal.shape[0]) // (nsegments + 1) 

    # the number of points for FFT should be larger than the segment length ...
    '''
    if nfft < segment_length:
        print("(nfft < nperseg): {0}, {1}".format(nfft,segment_length))
    '''
        
    print("nperseg",segment_length)
    
    from matplotlib import pyplot
    pyplot.plot(range(signal.shape[0]), signal, 'g')
    pyplot.title('Filtered green signal')
    pyplot.show()

    

    from scipy.signal import welch
    signal = signal.flatten()
    green_f, green_psd = welch(signal, framerate, 'flattop', nperseg=segment_length) #, scaling='spectrum',nfft=2048)
    print("Green F, Shape",green_f,green_f.shape)
    print("Green PSD, Shape",green_psd,green_psd.shape)

    #green_psd = green_psd.flatten()
    first = np.where(green_f > 0.9)[0] #0.8 for 300 frames
    last = np.where(green_f < 1.8)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    print("Range of interest",range_of_interest)
    max_idx = np.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]

    hr = f_max*60.0
    print("Heart rate = {0}".format(hr))

    import scipy.io as sio
    #mat_file_name = pulsedir + "pulse_" + video_file_name[:-4] + "_frame-0-15" + ".mat"
    mat_file_name = "pulse_" + video_file_name[:-4] + "_frame-0-15" + ".mat"
    sio.savemat(mat_file_name,{'pulse':signal, 'heartrate':hr, 'nperseg':segment_length})


    
    from matplotlib import pyplot
    pyplot.semilogy(green_f, green_psd, 'g')
    xmax, xmin, ymax, ymin = pyplot.axis()
    pyplot.vlines(green_f[range_of_interest[max_idx]], ymin, ymax, color='red')
    pyplot.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
    pyplot.show()
    

if __name__ == "__main__":
	main()
    
import numpy as np
import cv2 
import os
import sys
sys.path.append("../core/")
import data_processing_core as dpc
import json

import csv
import h5py
import math

# Prepare the following paths
root = "" # Input image folder
out_root = "" # Output image folder

h5_file_path = "GazeCapture_supplementary.h5"
gaze_label_name = "3d_gaze_target"
face_model_fpath = './sfm_face_coordinates.npy'
face_model_3d_coordinates = np.load(face_model_fpath)

def read_h5_file(path):
    f = h5py.File(path, "r")
    return f

#print(f["00002"]["file_name"][0])

def vector_to_yaw_pitch(x):
    # spherical_vector[0] = math.atan2(normalized_gaze[0],-normalized_gaze[2])
    # spherical_vector[1] = math.asin(normalized_gaze[1])
    x = np.reshape(x, (-1, 3))
    x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    output = np.zeros((x.shape[0], 2))
    output[:,0] = np.arctan2(x[:,0], - x[:,2])
    output[:,1] = np.arcsin(x[:,1])
    return output

def gazeto3d(gaze):
  assert gaze.size == 2, "The size of gaze must be 2"
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def ImageProcessing_GazeCapture():
    persons = os.listdir(root)
    persons.sort()

    length = len(persons)

    
    h5_data = read_h5_file(h5_file_path)
    with open("test.label", "w", newline="") as label_file_opened:
        writer = csv.DictWriter(label_file_opened, ["Face", "Left", "Right", "Origin", "3DGaze", "2DGaze"], delimiter=" ")#csv.writer(label_file_opened, delimiter=" ")
        writer.writeheader()
        for count, person in enumerate(persons):
            try:
                h5_person = h5_data[person]
            except:
                print("Person ", person, " does not exist!")
            im_root = os.path.join(root, person)
            try:
                person_info = json.load(open(os.path.join(im_root, "info.json")))
            except:
                continue
            splited_set = person_info["Dataset"]
            devices = person_info["DeviceName"]

            if person_info["Dataset"] != "test":
                continue
            
            
            im_outpath = os.path.join(out_root, "Image", "test")
            label_outpath = os.path.join(out_root, "Label", splited_set, f"{person}.label")

            if not os.path.exists(os.path.join(im_outpath, 'Face')):
                os.makedirs(os.path.join(im_outpath, 'Face'))

            if not os.path.exists(os.path.join(im_outpath, 'Left')):
                os.makedirs(os.path.join(im_outpath, 'Left'))

            if not os.path.exists(os.path.join(im_outpath, 'Right')):
                os.makedirs(os.path.join(im_outpath, 'Right'))

            #if not os.path.exists(os.path.join(im_outpath, 'grid')):
            #    os.makedirs(os.path.join(im_outpath, 'grid'))

            if not os.path.exists(os.path.join(out_root, "Label", splited_set)):
                os.makedirs(os.path.join(out_root, "Label", splited_set))

            progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count/length * 20))
            progressbar = "\r" + progressbar + f" {count}|{length}, Prcessing {person}.."
            print(progressbar, end="", flush=True)

            ImageProcessing_Person(h5_person, writer, im_root, im_outpath, label_outpath, person, devices)


def ImageProcessing_Person(h5_person, writer, im_root, im_outpath, label_outpath, person, devices):
    # Read annotation files
    frames = json.load(open(os.path.join(im_root, "frames.json")))
    face_located = json.load(open(os.path.join(im_root, "appleFace.json")))
    left_located = json.load(open(os.path.join(im_root, "appleLeftEye.json")))
    right_located = json.load(open(os.path.join(im_root, "appleRightEye.json")))
    grid_info = json.load(open(os.path.join(im_root, "faceGrid.json")))
    gt_info = json.load(open(os.path.join(im_root, "dotInfo.json")))

    outfile = open(label_outpath, 'w')
    outfile.write("Face Left Right Grid Xcam,Ycam Xdot,Ydot Device\n")

    for index, frame in enumerate(frames):
        try:
            gaze_target = h5_person['3d_gaze_target'][index, :].reshape(3, 1)
        except:
            print("Error index!")
            print(index, frame)
            continue

        # Calculate rotation matrix and euler angles
        rvec = h5_person['head_pose'][index, :3].reshape(3, 1)
        tvec = h5_person['head_pose'][index, 3:].reshape(3, 1)
        rotate_mat, _ = cv2.Rodrigues(rvec)

        landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
        landmarks_3d += tvec.T
        
        gaze_origin = np.mean(landmarks_3d[10:12, :], axis=0)  # between 2 eyes
        gaze_origin = gaze_origin.reshape(3, 1)
        gaze_vector = gaze_target - gaze_origin
        gaze_vector = gaze_vector/np.linalg.norm(gaze_vector)

        gaze_angle_rad = vector_to_yaw_pitch(gaze_vector)[0].tolist()
        
        # tmp = gaze_angle_rad[1]
        # gaze_angle_rad[1] = gaze_angle_rad[0]
        # gaze_angle_rad[0] = tmp

        gaze_vector = gaze_vector.T.tolist()

        #gaze_angle_rad = [math.radians(gaze_angle[0]), math.radians(gaze_angle[1])]

        if not face_located["IsValid"][index]: continue
        if not left_located["IsValid"][index]: continue
        if not right_located["IsValid"][index]: continue
        if not grid_info["IsValid"][index]: continue

        im_path = os.path.join(im_root, "frames", frame)
        img = cv2.imread(im_path)


        face_img = CropImg(img, face_located["X"][index], face_located["Y"][index], 
                                face_located["W"][index], face_located["H"][index])

        left_img = CropImg(face_img, left_located["X"][index], left_located["Y"][index], 
                                left_located["W"][index], left_located["H"][index])

        right_img = CropImg(face_img,right_located["X"][index],right_located["Y"][index], 
                            right_located["W"][index],right_located["H"][index])

        face_path = os.path.join(im_outpath, 'Face', person+"_"+frame).replace(os.sep, "/")
        left_eye = os.path.join(im_outpath, 'Left', person+"_"+frame).replace(os.sep, "/")
        right_eye = os.path.join(im_outpath, 'Right', person+"_"+frame).replace(os.sep, "/")
        #grid_frame = os.path.join(im_outpath, 'grid', frame).replace(os.sep, "/")
        # im_path
        cv2.imwrite(face_path, face_img)
        cv2.imwrite(left_eye, left_img)
        cv2.imwrite(right_eye, right_img)
        #cv2.imwrite(grid_frame, grid)
        
        data = {"Face": face_path, "Left": left_eye, "Right": right_eye, "Origin": im_path.replace(os.sep, "/"), 
                "3DGaze": str(list(gaze_vector)).strip("[]").replace(" ", ""), 
                "2DGaze": str(list(gaze_angle_rad)).strip("[]").replace(" ", "")
                }
        writer.writerow(data)

    outfile.close()

def CropImg(img, X, Y, W, H):
    Y_lim, X_lim, _ = img.shape
    H =  min(H, Y_lim)
    W = min(W, X_lim)

    X, Y, W, H = list(map(int, [X, Y, W, H]))
    X = max(X, 0)
    Y = max(Y, 0)

    if X + W > X_lim:
        X = X_lim - W

    if Y + H > Y_lim:
        Y = Y_lim - H

    return img[Y:(Y+H),X:(X+W)]

if __name__ == "__main__":
    ImageProcessing_GazeCapture()

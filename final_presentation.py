import time, cv2, sys, os, math
from threading import Thread
from djitellopy import Tello
import pygame
from pygame.locals import *
import numpy as np
import json
from Quadcopter.PID import FrontEnd


video_file="demo_3draft"
path= "C:/Users/65965/Desktop/School/FYPS2"
path_to_weights= f"{path}/yolov5/weights/v2_b6lab_teambuild.pt"
bim_checklist_path=r"C:\Users\65965\Desktop\School\FYPS2\revit\BIM_Checklist.csv"

"""Functions"""
def init_tello():
    tello.connect()
    batt=tello.get_battery()
    if int(batt)<20:
        print (" no battery", tello.get_battery())
        sys.exit(0)
    print(batt)

def videoRecorder():
    # create a VideoWrite object, recoring to ./video.avi
    height, width, _ = frame_read.frame.shape
    video = cv2.VideoWriter( f'{path}\DJITelloPy\elizabeth\drone_video_capture\{video_file}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    # time.sleep(5)
    while keepRecording:
        try:
            frame= cv2.cvtColor(frame_read.frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
            time.sleep(1 / 30)
            # print(frame)
        except KeyboardInterrupt:
            # keepRecording = False
            sys.exit(0)
    video.release()

def yolov5():
    cmd= f"python {path}/yolov5/detect.py --weights {path_to_weights} --img 416 --conf 0.5 --source {path}\DJITelloPy\elizabeth\drone_video_capture\{video_file}.avi --device 0 --save-txt --name result_{video_file} "
    os.system(cmd)

def deepsort():
    cmd= f"python c:/Users/65965/Desktop/School/FYPS2/deepsort/track.py --yolo_model {path_to_weights} --source {path}\DJITelloPy\elizabeth\drone_video_capture\{video_file}.avi --device 0 --save-txt --name result_{video_file} --conf-thres 0.5 --device 0 --show-vid --name result_{video_file} --bim_checklist {bim_checklist_path}"
    os.system(cmd)

def mission_pad_landing():
    """mission pad landing"""
    """
    pad = tello.get_mission_pad_id()
    print("mission pad ", pad)
    # tello.move_down(20)

    for i in range(1):
        for i in range (10):
            if pad != 1:
                print("mission pad not found, trying")
                time.sleep(5)
                tello.send_command_with_return("command",5)
                pad = tello.get_mission_pad_id()
        print ("took ",i, " tries")
        print("mission pad found")
        time.sleep(1)
        tello.go_xyz_speed_mid(0,0,20,10,1)
        print("sleep")
        time.sleep(5)
        """

def mission_pad_takeoff():
    """mission pad landing"""
    """
    tello.enable_mission_pads()
    tello.set_mission_pad_detection_direction(2) 

    tello.takeoff()
    time.sleep(1)

    print(tello.get_mission_pad_id())
    for i in range (0,4):
        pad = tello.get_mission_pad_id()
        if pad==1:
            tello.go_xyz_speed_mid(0,0,150,10,1)
            print("m1 found")
            time.sleep(5)
            break
        print("try" , i)
    tello.move_up(100)
    """

def rotate():
    time.sleep(2)
    for i in range(12):
        time.sleep(0.1)
        tello.rotate_clockwise(30)
        time.sleep(0.5)
    tello.rotate_clockwise(15)
    
def move_d1( d1_dir ):
    """Go by D1"""
    if 20 <= d1 <= 500 :
        if d1_dir == 0 :
            tello.move_forward(d1)
        elif d1_dir == 1:
            tello.move_back(d1)
        time.sleep(0.5)
    else: print("out of range, skipping d1")

def move_d2( d2_dir ):
    """Go by D2"""
    if 20 <= d2 <= 500 :
        if d2_dir == 0 :
            tello.move_left(d2)
        elif d2_dir == 1:
            tello.move_right(d2)
        time.sleep(0.5)
    else: print("out of range, skipping d2")


# """Start Tello"""
tello = Tello()
init_tello()

""""Ensure frame and start recording"""
tello.streamon()
# time.sleep(2)
tello.send_command_with_return("downvision 1")
frame_read = tello.get_frame_read()
print('initialising frame_read')
time.sleep(5)
tello.send_command_with_return("command")
time.sleep(5)
print(frame_read)
"""Wait for JSON Command to start"""
while(True):
    # scp.get('scout_drone_data.json')

    with open('scout_drone_data.json') as f:
        data = json.load(f)
        f.close()

    if(data['takeoff_drone_flag'] == 1 and data['drone_landed_flag'] == 0):
        """Load Coordinates"""
        x1 = float(data['takeoff_drone_x1'])*100
        y1 = float(data['takeoff_drone_y1'])*100
        x2 = float(data['takeoff_drone_x2'])*100
        y2 = float(data['takeoff_drone_y2'])*100
        x3 = float(data['takeoff_drone_x3'])*100
        y3 = float(data['takeoff_drone_y3'])*100 
        d1_dir = int(data['takeoff_drone_d1_dir']) #left--(0), right--(1)
        d2_dir = int(data['takeoff_drone_d2_dir']) #forward--(0), backward--(1)

        d1= int(math.sqrt(((x2-x1)**2)+((y2-y1)**2)))
        d2= int(math.sqrt(((x3-x2)**2)+((y3-y2)**2)))

        print(d1)
        print(d2)
        print("initialising drone takeoff...")
        break
    else:
        time.sleep(1)
        tello.send_command_with_return("command")
        print("Waiting for scout's command...")

print(frame_read.frame, f"moving by {d1}, {d2}")
# """"takeoff"""
tello.takeoff()
tello.move_up(30)
frontend = FrontEnd(tello,False)
frontend.run()
tello.set_speed(30)
"""Move to coordinates"""
move_d1(d1_dir)
move_d2(d2_dir)


"""Switch to Front Camera, Start recorder"""
tello.send_command_with_return("downvision 0")
tello.send_command_with_return("downvision 0")

frame_read = tello.get_frame_read()
print(frame_read.frame)

keepRecording = True
recorder = Thread(target=videoRecorder)
recorder.start()

"""Rotate while Recording"""
rotate()
"""End Recording"""
keepRecording = False
time.sleep(1)

"""Return to Aruco"""
move_d2(d2_dir^1)
move_d1(d1_dir^1)
tello.move_up(20)

"""Land Tello"""
frontend = FrontEnd(tello, True)
frontend.run()
print(tello.get_battery())

# """Stop Recording"""
recorder.join()
print(f"video saved to {path}\DJITelloPy\elizabeth\drone_video_capture\{video_file}.avi" )

# # data['drone_landed_flag'] = 1
# # data['takeoff_drone_flag'] = 0

# # with open('scout_drone_data.json','w') as f:
# #     data = json.dump(data,f)
# #     f.close()

# """Start Detections"""
# deepsort = Thread(target=deepsort)
# deepsort.start()
deepsort()
tello.end()
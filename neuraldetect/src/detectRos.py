#!/usr/bin/env python
import rospy
import std_msgs.msg
from ctypes import POINTER, c_float
import math
import cv2
import numpy
from ultralytics import YOLO
import pyzed.sl as sl



# Load a model
#model = YOLO("yolov8x-seg.pt")  # load an official model
model = YOLO(argv[0])  # load a custom model


zed = sl.Camera()

# Create a InitParameters object and set configuration parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.ULTRA    # Use PERFORMANCE depth mode
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_maximum_distance = 40  # Use meter units (for depth measurements)
init_params.camera_resolution = sl.RESOLUTION.HD720


err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
# Setting the depth confidence parameters
runtime_parameters.confidence_threshold = 100
runtime_parameters.textureness_confidence_threshold = 100

image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()

mirror_ref = sl.Transform()
mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
tr_np = mirror_ref.m

def callback(msg):
    print(msg)


rospy.init_node("nerualDetect")
rospy.loginfo('start')
sub = rospy.Subscriber("sub", std_msgs.msg.String, callback)
pub = rospy.Publisher('pub', std_msgs.msg.Int16, queue_size=10)
rate = rospy.Rate(1)
while not rospy.is_shutdown():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            image_ocv = image.get_data()
            img = cv2.cvtColor(image_ocv, cv2.COLOR_BGRA2BGR)
            results = model(img)
            ploted = results[0].plot()
            for res in results:
                for box in res.boxes.xyxy:
                        x1 =box[0].item()
                        y1 =box[1].item()
                        x2 =box[2].item()
                        y2 =box[3].item()
                        
                        err, point_cloud_value = point_cloud.get_value(int((x1+x2)/2),int((y1+y2)/2))
                        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                 point_cloud_value[1] * point_cloud_value[1] +
                                 point_cloud_value[2] * point_cloud_value[2])
                        ploted = cv2.circle(ploted, (int((x1+x2)/2),int((y1+y2)/2)), radius=2, color=(0, 0, 255), thickness=3)
                        #cv2.putText(ploted, str(round(distance,3)), (int((x1+x2)/2),int((y1+y2)/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 3)
                        pub.publish(3)

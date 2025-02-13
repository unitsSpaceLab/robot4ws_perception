#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import struct
import ros_numpy
from robot4ws_msgs.msg import ColorMessage
from nav_msgs.msg import Odometry


class ColorDetectorWithDepth:
    def __init__(self):
        rospy.init_node('color_detector_with_depth', anonymous=True)

        self.bridge = CvBridge()
        self.load_parameters()

        self.max_depth_range = 2500 # mm
        self.min_depth_range = 100 # mm
        
        if self.debug_visualization:
            self.debug_publisher = rospy.Publisher("/color_detection/debug_image", Image, queue_size=1)

        # Synchronize odom and color, depth images
        self.odom_sub = Subscriber(self.odom_topic_name, Odometry)
        self.color_sub = Subscriber(self.color_topic, Image)
        self.depth_sub = Subscriber(self.depth_topic, Image)
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.odom_sub],
            queue_size=1,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

        # Camera info subscriber
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        self.camera_info = None
        self.color_msg_header = None

        self.pc2_publisher = rospy.Publisher(self.color_detection_topic_name, ColorMessage, queue_size=1)
        self.pc2_debug_publisher = rospy.Publisher("pc2_color_debug", PointCloud2, queue_size=1)

        # Visualization colors
        self.viz_colors = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0)
        }

        self.viz_colors_uint32 = {
            color_name: (bgr[0] << 16) | (bgr[1] << 8) | bgr[2]
            for color_name, bgr in self.viz_colors.items()
        }

        self.point_dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('bgr', np.uint32)
        ])

    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        self.color_topic = rospy.get_param('camera_topics/color')
        self.depth_topic = rospy.get_param('camera_topics/depth')
        self.camera_info_topic = rospy.get_param('camera_topics/camera_info')
        self.odom_topic_name = rospy.get_param('odom_topic_name')
        self.color_detection_topic_name = rospy.get_param('color_detection_topic_name')
        
        # Get color definitions
        self.colors = {}
        colors_param = rospy.get_param('colors')
        for color_name, color_data in colors_param.items():
            self.colors[color_name] = {
                'lower': np.array(color_data['hsv_lower']),
                'upper': np.array(color_data['hsv_upper']),
                'name': color_data['name']
            }
        
        # Get processing parameters
        self.min_contour_area = rospy.get_param('processing/min_contour_area', 100)
        self.blur_kernel_size = rospy.get_param('processing/blur_kernel_size', 5)
        self.morph_kernel_size = rospy.get_param('processing/morph_kernel_size', 3)
        self.debug_visualization = rospy.get_param('processing/debug_visualization', True)

    # Not used 
    def detect_color(self, hsv_image, color_name):
        """Detect specific color in the image"""
        color_params = self.colors[color_name]
        mask = cv2.inRange(hsv_image, color_params['lower'], color_params['upper'])
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        return valid_contours

    def camera_info_callback(self, msg):
        self.camera_info = msg

        self.color_msg_header = rospy.Header()
        self.color_msg_header.frame_id = msg.header.frame_id

        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]

        self.camera_info_sub.unregister()

    def preprocess_image(self, image):
        """Apply preprocessing to the image"""
        blurred = cv2.GaussianBlur(image, (self.blur_kernel_size, self.blur_kernel_size), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv
    
    def image_callback(self, color_msg, depth_msg, odom):
        """Process incoming synchronized color and depth images with odom"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            depth_image_filtered = cv2.medianBlur(depth_image, 5)

            hsv_image = self.preprocess_image(cv_image)

            points = []

            for color_name in self.colors.keys():
                valid_contours = self.detect_color(hsv_image, color_name)

                bgr_uint32 = self.viz_colors_uint32[color_name]

                full_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

                cv2.drawContours(full_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

                valid_pixels = np.where(
                    (full_mask != 0) & 
                    (depth_image_filtered < self.max_depth_range) &
                    (depth_image_filtered > self.min_depth_range)
                )

                for cy, cx in zip(*valid_pixels):
                    depth = depth_image_filtered[cy, cx]
                    point = self.depth_to_3d(cx, cy, depth)
                    points.append((point[0], point[1], point[2], bgr_uint32))

            if len(points) > 0:
                self.publish_color_msg(points, odom, color_msg.header)

            # Debug visualization (se necessario)
            # if self.debug_visualization:
            #     debug_image = self.create_debug_image(cv_image, all_detections)
            #     debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            #     self.debug_publisher.publish(debug_msg)

        except CvBridgeError as e:
            rospy.logerr("CV Bridge error: {}".format(e))
        except Exception as e:
            rospy.logerr("Error processing images: {}".format(e))


    def publish_color_msg(self, points, odom, header):
        """Publish a PointCloud2 message"""
        # Header per il messaggio PointCloud2
        self.color_msg_header.stamp = rospy.Time.now()

        structured_points = np.array(points, dtype=self.point_dtype)

        cloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(structured_points)

        cloud_msg.header = header #self.color_msg_header

        self.pc2_debug_publisher.publish(cloud_msg)

        msg = ColorMessage()

        msg.header = self.color_msg_header
        msg.color_cloud = cloud_msg
        msg.odom = odom

        self.pc2_publisher.publish(msg)

    def depth_to_3d(self, u, v, depth):
        """Convert 2D pixel (u, v) and depth value to 3D coordinates"""
        
        depth /= 1000.0 # mm -> m 
        
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return np.array([x, y, z])

    def create_debug_image(self, image, all_detections):
        debug_image = image.copy()
        for color_name, contours in all_detections.items():
            viz_color = self.viz_colors[color_name]
            cv2.drawContours(debug_image, contours, -1, viz_color, 2)
            for contour in contours:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(debug_image, color_name, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, viz_color, 2)
        return debug_image

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ColorDetectorWithDepth()
        detector.run()
    except rospy.ROSInterruptException:
        pass

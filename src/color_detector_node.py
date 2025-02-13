#!/usr/bin/env python

from __future__ import division
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from robot4ws_msgs.msg import ColorDetection3D, ColorDetection3DArray
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from visualization_msgs.msg import Marker, MarkerArray 
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from nav_msgs.msg import Odometry

class ColorDepthDetector(object):
    def __init__(self):
        rospy.init_node('color_depth_detector', anonymous=True)
        self._setup_tf()
        self._setup_bridge()
        self._load_parameters()
        self._setup_publishers()
        self._setup_subscribers()
        
    def _setup_tf(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
    def _setup_bridge(self):
        self.bridge = CvBridge()

    def _load_parameters(self):
        # Core detection parameters
        self.max_distance = rospy.get_param('processing/max_distance', 2.5)  # meters
        self.min_distance = rospy.get_param('processing/min_distance', 0.3)  # meters
        self.max_object_size = rospy.get_param('processing/max_object_size', 0.8)  # meters
        self.min_object_size = rospy.get_param('processing/min_object_size', 0.02)  # meters
        self.depth_sample_size = rospy.get_param('processing/depth_sample_size', 3)
        self.min_confidence = rospy.get_param('processing/min_confidence', 0.3)
        
        # Basic processing parameters
        self.min_contour_area = rospy.get_param('processing/min_contour_area', 100)
        self.max_contour_area = rospy.get_param('processing/max_contour_area', float('inf'))
        self.blur_kernel_size = rospy.get_param('processing/blur_kernel_size', 5)
        self.blur_kernel_size_large = rospy.get_param('processing/blur_kernel_size_large', 15)
        self.morph_kernel_size = rospy.get_param('processing/morph_kernel_size', 3)
        self.morph_kernel_size_large = rospy.get_param('processing/morph_kernel_size_large', 15)
        
        # Surface area thresholds
        self.large_surface_threshold = rospy.get_param('processing/large_surface_threshold', 5000)
        
        # Color definitions (simplified)
        self.colors = {
            'red': {
                'lower': np.array([0,100,100]), 
                'upper': np.array([10,255,255]), 
                'rgba': (1.0,0.0,0.0,0.5),
                'rgb': [255,0,0]
            },
            'blue': {
                'lower': np.array([100,100,100]), 
                'upper': np.array([130,255,255]), 
                'rgba': (0.0,0.0,1.0,0.5),
                'rgb': [0,0,255]
            },
            'green': {
                'lower': np.array([40,100,100]), 
                'upper': np.array([80,255,255]), 
                'rgba': (0.0,1.0,0.0,0.5),
                'rgb': [0,255,0]
            },
            'yellow': {
                'lower': np.array([20,100,100]), 
                'upper': np.array([30,255,255]), 
                'rgba': (1.0,1.0,0.0,0.5),
                'rgb': [255,255,0]
            },
            'purple': {
                'lower': np.array([130,100,100]), 
                'upper': np.array([150,255,255]), 
                'rgba': (0.5,0.0,0.5,0.5),
                'rgb': [128,0,128]
            },
            'cyan': {
                'lower': np.array([85,100,100]), 
                'upper': np.array([95,255,255]), 
                'rgba': (0.0,1.0,1.0,0.5),
                'rgb': [0,255,255]
            },
            'orange': {
                'lower': np.array([10,100,100]), 
                'upper': np.array([20,255,255]), 
                'rgba': (1.0,0.65,0.0,0.5),
                'rgb': [255,165,0]
            }
        }

    def _setup_publishers(self):
        self.detection3d_pub = rospy.Publisher("/color_detection/detections3d", ColorDetection3DArray, queue_size=1)
        self.marker_pub = rospy.Publisher("/color_detection/markers", MarkerArray, queue_size=1)

    def _setup_subscribers(self):
        self.color_sub = Subscriber("/Archimede/d435i_camera/color/image_raw", Image)
        self.depth_sub = Subscriber("/Archimede/d435i_camera/depth/image_raw", Image)
        self.camera_info_sub = Subscriber("/Archimede/d435i_camera/color/camera_info", CameraInfo)
        self.odom_sub = Subscriber("gazebo_2_odom", Odometry)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.camera_info_sub, self.odom_sub],
            queue_size=5,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

    def get_3d_point(self, depth_image, x, y, camera_info, is_large_surface=False):
        # Sample multiple depth points around the target
        x, y = int(x), int(y)
        k = self.depth_sample_size * 2 if is_large_surface else self.depth_sample_size
        k = k // 2
        depth_window = depth_image[max(0, y-k):min(depth_image.shape[0], y+k+1),
                                 max(0, x-k):min(depth_image.shape[1], x+k+1)]
        
        valid_depths = depth_window[depth_window > 0]
        min_valid_points = 8 if is_large_surface else 4
        if len(valid_depths) < min_valid_points:
            return None
            
        depth_value = float(np.median(valid_depths))
        depth_meters = depth_value * 0.001  # mm to meters
        
        if not self.min_distance <= depth_meters <= self.max_distance:
            return None
            
        # Camera intrinsics for point projection
        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]
        
        x_meters = (x - cx) * depth_meters / fx
        y_meters = (y - cy) * depth_meters / fy
        
        return Point(x=x_meters, y=y_meters, z=depth_meters)

    #TODO: update with static tf
    def transform_point_to_world(self, point, header):
        try:
            transform = self.tf_buffer.lookup_transform(
                # 'Archimede_foot_start',
                'Archimede_footprint',
                header.frame_id,
                header.stamp,
                rospy.Duration(0.1)
            )

            pose_stamped = PoseStamped()
            pose_stamped.header = header
            pose_stamped.pose.position = point
            pose_stamped.pose.orientation = Quaternion(0, 0, 0, 1)

            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            return transformed_pose.pose.position, transform.transform.rotation

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Transform failed: {0}".format(e))
            return None, None

    def _calculate_confidence(self, contour_area, depth, bbox_size, is_large_surface=False):
        if is_large_surface:
            area_score = min(contour_area / (self.large_surface_threshold * 2.0), 1.0)
        else:
            area_score = min(contour_area / 5000.0, 1.0)
            
        depth_score = max(0, 1.0 - (depth / self.max_distance))
        size_score = max(0, 1.0 - (max(bbox_size) / self.max_object_size))
        
        # Give more weight to area for large surfaces
        if is_large_surface:
            return (area_score * 2.0 + depth_score + size_score) / 4.0
        return (area_score + depth_score + size_score) / 3.0

    def create_detection3d_msg(self, contour, depth_image, color_name, header, camera_info):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None
            
        contour_area = cv2.contourArea(contour)
        is_large_surface = contour_area > self.large_surface_threshold
        
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        point = self.get_3d_point(depth_image, cx, cy, camera_info, is_large_surface)
        if point is None:
            return None
            
        world_point, orientation = self.transform_point_to_world(point, header)
        if world_point is None:
            return None
            
        bbox_x, bbox_y, w, h = cv2.boundingRect(contour)
        scale_factor = 0.001 * point.z
        
        if is_large_surface:
            min_size = self.min_object_size * 2.0
            bbox_size = [
                min(max(float(w) * scale_factor, min_size), self.max_object_size * 2.0),
                min(max(float(h) * scale_factor, min_size), self.max_object_size * 2.0),
                min(max(0.2, float(min(w, h)) * scale_factor), self.max_object_size)
            ]
        else:
            bbox_size = [
                min(max(float(w) * scale_factor, self.min_object_size), self.max_object_size),
                min(max(float(h) * scale_factor, self.min_object_size), self.max_object_size),
                min(max(0.1, float(min(w, h)) * scale_factor), self.max_object_size)
            ]
            
        confidence = self._calculate_confidence(contour_area, point.z, bbox_size, is_large_surface)
        if confidence < self.min_confidence:
            return None
            
        # Create Detection3D
        detection = Detection3D()
        detection.header = header
        detection.header.frame_id = header.frame_id
        detection.bbox.center.position = world_point
        detection.bbox.center.orientation = orientation
        detection.bbox.size.x = bbox_size[0]
        detection.bbox.size.y = bbox_size[1]
        detection.bbox.size.z = bbox_size[2]
        
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.score = confidence
        detection.results.append(hypothesis)
        
        # Create ColorDetection3D
        color_detection = ColorDetection3D()
        color_detection.detection = detection
        color_detection.color_name = color_name
        
        # Add contour points for large surfaces
        contour_points = []
        if is_large_surface:
            for point in contour:
                x, y = point[0][0], point[0][1]
                point3d = self.get_3d_point(depth_image, x, y, camera_info)
                if point3d:
                    world_point_contour, _ = self.transform_point_to_world(point3d, header)
                    if world_point_contour:
                        contour_points.append(world_point_contour)
            color_detection.contour_points = contour_points
            
            # Adjust RGB values for large surfaces
            rgb = self.colors[color_name]['rgb']
            color_detection.color_rgb = [min(255, int(val * 1.2)) for val in rgb]
        else:
            color_detection.color_rgb = self.colors[color_name]['rgb']
        
        return color_detection, is_large_surface
        
    def _detect_color(self, hsv_image, color_params):
        # Check for large surfaces first
        mask_large = cv2.inRange(hsv_image, color_params['lower'], color_params['upper'])
        if self.morph_kernel_size_large > 0:
            kernel_large = np.ones((self.morph_kernel_size_large, self.morph_kernel_size_large), np.uint8)
            mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_OPEN, kernel_large)
            mask_large = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel_large)
        
        # Regular detection
        mask = cv2.inRange(hsv_image, color_params['lower'], color_params['upper'])
        if self.morph_kernel_size > 0:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours_large, _ = cv2.findContours(mask_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # Combine contours, filtering by area
        all_contours = []
        for cnt in contours_large + contours:
            area = cv2.contourArea(cnt)
            if self.min_contour_area < area < self.max_contour_area:
                all_contours.append(cnt)
                
        return all_contours

    def _create_marker(self, detection, color, marker_id, is_large_surface=False):
        marker = Marker()
        marker.header.frame_id = "Archimede_foot_start"
        marker.header.stamp = detection.header.stamp
        marker.ns = "color_detections"
        marker.id = marker_id
        
        # Use different marker types for large surfaces
        if is_large_surface:
            marker.type = Marker.CUBE  # Use cube for large surfaces
            marker.scale = detection.bbox.size  # Keep original scale for large surfaces
        else:
            marker.type = Marker.CYLINDER
            marker.scale = detection.bbox.size
            
        marker.action = Marker.ADD
        marker.pose.position = detection.bbox.center.position
        marker.pose.orientation = detection.bbox.center.orientation
        
        # Adjust visualization for large surfaces
        r, g, b, a = color
        if is_large_surface:
            a = a * 0.7  # More transparent
            # Make marker slightly bigger for large surfaces
            marker.scale.x *= 1.2
            marker.scale.y *= 1.2
            marker.scale.z *= 1.2
        
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        return marker

    def image_callback(self, color_msg, depth_msg, info_msg, odom_msg):
        try:
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            
            # Pre-process image with both kernel sizes
            blurred_large = cv2.GaussianBlur(cv_color, 
                                           (self.blur_kernel_size_large, self.blur_kernel_size_large), 
                                           0)
            blurred = cv2.GaussianBlur(cv_color, 
                                      (self.blur_kernel_size, self.blur_kernel_size), 
                                      0)
            
            hsv_image_large = cv2.cvtColor(blurred_large, cv2.COLOR_BGR2HSV)
            hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            # Create ColorDetection3DArray
            detection_array = ColorDetection3DArray()
            detection_array.header = color_msg.header
            detection_array.header.stamp = color_msg.header.stamp
            detection_array.header.frame_id = color_msg.header.frame_id
            detection_array.odom = odom_msg
            
            marker_array = MarkerArray()
            
            for color_name, color_params in self.colors.iteritems():
                # First try with large surface parameters
                contours = self._detect_color(hsv_image_large, color_params)
                
                # If no large contours found, try with regular parameters
                if not contours or all(cv2.contourArea(cnt) <= self.large_surface_threshold for cnt in contours):
                    contours = self._detect_color(hsv_image, color_params)
                
                for contour in contours:
                    detection_result = self.create_detection3d_msg(
                        contour, cv_depth, color_name, color_msg.header, info_msg
                    )
                    
                    if detection_result:
                        detection, is_large_surface = detection_result
                        detection_array.detections.append(detection)
                        marker = self._create_marker(
                            detection.detection, 
                            color_params['rgba'], 
                            len(marker_array.markers),
                            is_large_surface
                        )
                        marker_array.markers.append(marker)
            
            if detection_array.detections:
                self.detection3d_pub.publish(detection_array)
                self.marker_pub.publish(marker_array)
                    
        except CvBridgeError as e:
            rospy.logerr("CV Bridge error: {0}".format(e))
        except Exception as e:
            rospy.logerr("Error processing image: {0}".format(e))

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ColorDepthDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
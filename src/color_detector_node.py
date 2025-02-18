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
        self.max_distance = rospy.get_param('processing/max_distance', 5.0)  # meters
        self.min_distance = rospy.get_param('processing/min_distance', 0.2)  # meters
        self.max_object_size = rospy.get_param('processing/max_object_size', 0.8)  # meters
        self.min_object_size = rospy.get_param('processing/min_object_size', 0.02)  # meters
        self.depth_sample_size = rospy.get_param('processing/depth_sample_size', 3)
        self.min_confidence = rospy.get_param('processing/min_confidence', 0.2)
        
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
        self.debug_image_pub = rospy.Publisher("/color_detection/debug_image", Image, queue_size=1)

    def _setup_subscribers(self):
        self.color_sub = Subscriber("/Archimede/d435i_camera/color/image_raw", Image)
        self.depth_sub = Subscriber("/Archimede/d435i_camera/depth/image_raw", Image)
        self.odom_sub = Subscriber("gazebo_2_odom", Odometry)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.odom_sub],
            queue_size=5,
            slop=0.1
        )
        self.sync.registerCallback(self.image_callback)

    def get_3d_point(self, depth_image, x, y):
        x, y = int(x), int(y)
        k = self.depth_sample_size
        
        # Create circular mask for sampling
        mask = np.zeros((2*k+1, 2*k+1), dtype=np.uint8)
        cv2.circle(mask, (k,k), k, 1, -1)
        
        y_start = max(0, y-k)
        y_end = min(depth_image.shape[0], y+k+1)
        x_start = max(0, x-k)
        x_end = min(depth_image.shape[1], x+k+1)
        
        actual_mask = mask[0:y_end-y_start, 0:x_end-x_start]
        depth_window = depth_image[y_start:y_end, x_start:x_end]
        
        valid_depths = depth_window[(depth_window > 100) & (depth_window < 10000) & (actual_mask == 1)]
        
        if len(valid_depths) < 5:
            return None
        
        if np.std(valid_depths) > 200:  
            return None
            
        # Use median depth for better noise immunity
        median_depth = np.median(valid_depths)
        depth_meters = median_depth * 0.001
        
        if not self.min_distance <= depth_meters <= self.max_distance:
            return None

        fx = 462.1379699707031
        fy = 462.1379699707031
        cx = 320.0
        cy = 240.0
        
        x_meters = (x - cx) * depth_meters / fx
        y_meters = (y - cy) * depth_meters / fy
        
        return Point(x=x_meters, y=y_meters, z=depth_meters)



    def _calculate_surface_properties(self, contour, depth_image):
        # Get bounding rect of the contour
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            return None

        # Create mask within bounding rect
        mask = np.zeros((h, w), dtype=np.uint8)
        adjusted_contour = contour - (x, y)
        cv2.drawContours(mask, [adjusted_contour], -1, 255, -1)

        # Extract depth ROI and valid depths
        depth_roi = depth_image[y:y+h, x:x+w]
        valid_mask = (depth_roi > 100) & (depth_roi < 10000) & (mask == 255)
        valid_depths = depth_roi[valid_mask]

        if len(valid_depths) < 5:
            return None

        # Calculate depth variance
        depth_variance = np.var(valid_depths) if len(valid_depths) > 1 else 0

        # Calculate gradient magnitudes within ROI
        grad_x = self.sobelx[y:y+h, x:x+w][valid_mask]
        grad_y = self.sobely[y:y+h, x:x+w][valid_mask]
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(grad_mag) if len(grad_mag) > 0 else 0

        # Aspect ratio of bounding rect
        aspect_ratio = float(w) / h if h != 0 else 0

        return {
            'area': cv2.contourArea(contour),
            'aspect_ratio': aspect_ratio,
            'avg_gradient': avg_gradient,
            'depth_variance': depth_variance
        }

    def is_ground_surface(self, surface_props, position_3d):
        if surface_props is None or position_3d is None:
            return False
        
        # Check if close to robot base
        height_threshold = 0.3  # meters
        is_near_base = abs(position_3d.z) < height_threshold
        
        # Surface properties checks
        is_flat = surface_props['avg_gradient'] < 250
        is_large = surface_props['area'] > 3000
        is_wide = surface_props['aspect_ratio'] > 1.5  
        
        confidence = (1.0 - abs(position_3d.z)/height_threshold) * 0.4 + \
                    (1.0 - surface_props['avg_gradient']/250) * 0.3 + \
                    min(surface_props['area']/6000, 1.0) * 0.3
        
        
        return is_near_base and is_flat and is_large and is_wide and confidence > 0.6


    def transform_point_to_world(self, point, header):
        try:
            transform = self.tf_buffer.lookup_transform(
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

    def _merge_nearby_contours(self, contours, distance_threshold=10):
        if not contours:
            return []
            
        # Calculate all centers at once
        centers = []
        valid_contours = []
        
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0:  # Only process valid contours
                centers.append((int(M['m10']/M['m00']), int(M['m01']/M['m00'])))
                valid_contours.append(cnt)
        
        if not centers:
            return []
            
        # Convert to numpy array for vectorized operations
        centers = np.array(centers)
        
        # Calculate all pairwise distances at once using broadcasting
        distances = np.sqrt(np.sum((centers[:, None] - centers) ** 2, axis=2))
        
        # Find groups of nearby contours
        merged = []
        used = set()
        
        for i in range(len(valid_contours)):
            if i in used:
                continue
                
            # Find all contours close to current one
            close_indices = np.where(distances[i] < distance_threshold)[0]
            
            if len(close_indices) > 1:  # If there are other close contours
                # Merge all close contours
                to_merge = [valid_contours[j] for j in close_indices if j not in used]
                if to_merge:
                    merged.append(np.concatenate(to_merge))
                    used.update(close_indices)
            elif i not in used:
                merged.append(valid_contours[i])
        
        return merged

    def _calculate_confidence(self, depth, bbox_size, surface_props, is_ground):
        if is_ground:
            # Ground confidence based on area and flatness
            area_norm = min(surface_props['area'] / 60000.0, 1.0)
            flatness = 1.0 - min(surface_props['avg_gradient'] / 500.0, 1.0)
            depth_score = np.exp(-0.3 * (depth / self.max_distance))  # Added mild depth penalty
            
            return min(area_norm * 0.4 + flatness * 0.4 + depth_score * 0.2, 1.0)
        else:
            # Object confidence factors
            depth_score = np.exp(-0.5 * (depth / self.max_distance))
            
            # Size consistency score (expected size vs actual)
            expected_size = 0.3  # Expected object size at 1m distance
            actual_size = bbox_size[0] * (depth / 1.0)  # Scale size to 1m reference distance
            size_consistency = max(0.0, 1.0 - abs(actual_size - expected_size) / expected_size)
            
            # Boundary sharpness score
            edge_sharpness = 1.0 - min(surface_props['avg_gradient'] / 1000.0, 1.0)
            
            # Combine factors with depth being most important
            confidence = min(
                0.5 * depth_score +
                0.3 * size_consistency +
                0.2 * edge_sharpness,
                1.0
            )
            
            return max(confidence, 0.1)  # Ensure minimum confidence of 0.1

    def create_detection3d_msg(self, contour, depth_image, color_name, header):
        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None

        # Get centroid first
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        # Compute 3D position early
        point = self.get_3d_point(depth_image, cx, cy)
        if point is None:
            return None
            
        world_point, orientation = self.transform_point_to_world(point, header)
        if world_point is None:
            return None

        # Now calculate surface properties
        surface_props = self._calculate_surface_properties(contour, depth_image)
        if surface_props is None:
            return None
            
        # Ground detection with 3D position context
        is_ground = self.is_ground_surface(surface_props, world_point)
        
        # Bounding box calculations
        bbox_x, bbox_y, w, h = cv2.boundingRect(contour)
        scale_factor = 0.001 * point.z
        bbox_size = [
            min(max(float(w) * scale_factor, self.min_object_size), self.max_object_size),
            min(max(float(h) * scale_factor, self.min_object_size), self.max_object_size),
            min(max(0.1, float(min(w, h)) * scale_factor), self.max_object_size)
        ]

        # Calculate confidence using 3D information
        confidence = self._calculate_confidence(
            depth=point.z,
            bbox_size=bbox_size,
            surface_props=surface_props,
            is_ground=is_ground
        )

        # Create Detection3D message
        detection = Detection3D()
        detection.header = header
        detection.bbox.center.position = world_point
        detection.bbox.center.orientation = orientation
        detection.bbox.size.x = bbox_size[0]
        detection.bbox.size.y = bbox_size[1]
        detection.bbox.size.z = bbox_size[2]
        
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.score = confidence
        detection.results.append(hypothesis)

        # Create ColorDetection3D message
        color_detection = ColorDetection3D()
        color_detection.detection = detection
        color_detection.color_name = color_name
        color_detection.is_ground = is_ground

        # Add contour points only for ground surfaces
        if is_ground:
            contour_points = []
            for point in contour:
                x, y = point[0][0], point[0][1]
                point3d = self.get_3d_point(depth_image, x, y)
                if point3d:
                    world_point_contour, _ = self.transform_point_to_world(point3d, header)
                    if world_point_contour:
                        contour_points.append(world_point_contour)
            color_detection.contour_points = contour_points

        color_detection.color_rgb = self.colors[color_name]['rgb']
        return color_detection, is_ground

    def _detect_color(self, hsv_image, color_params):
        """Optimized version of color detection"""

        small_hsv = cv2.resize(hsv_image, (hsv_image.shape[1]//4, hsv_image.shape[0]//4))
        small_mask = cv2.inRange(small_hsv, color_params['lower'], color_params['upper'])
        
        small_contours, _ = cv2.findContours(small_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        if not small_contours:
            return []
            
        all_contours = []
        for small_cnt in small_contours:
            # Scale up the bounding box
            x, y, w, h = cv2.boundingRect(small_cnt)
            x, y, w, h = x*4, y*4, w*4, h*4
            
            # Add padding
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(hsv_image.shape[1] - x, w + 2*pad)
            h = min(hsv_image.shape[0] - y, h + 2*pad)
            
            # Process only the ROI
            roi = hsv_image[y:y+h, x:x+w]
            mask = cv2.inRange(roi, color_params['lower'], color_params['upper'])
            
            # Adaptive kernel size based on ROI size
            kernel_size = min(max(3, min(w, h) // 50), 15)
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
                
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Single morphological operation combining open and close effects
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours in ROI
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            
            for cnt in contours:
                cnt += [x, y]
                area = cv2.contourArea(cnt)
                if self.min_contour_area < area < self.max_contour_area:
                    all_contours.append(cnt)
        
        if len(all_contours) > 1:
            # Use distance matrix to merge contours
            centers = []
            for cnt in all_contours:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    centers.append((cx, cy))
            
            # Create distance matrix
            centers = np.array(centers)
            if len(centers) > 0:
                distances = np.sqrt(((centers[:, None] - centers) ** 2).sum(axis=2))
                merge_threshold = 10
                
                # Find contours to merge
                merged = []
                used = set()
                for i in range(len(all_contours)):
                    if i in used:
                        continue
                        
                    to_merge = [all_contours[i]]
                    for j in range(i + 1, len(all_contours)):
                        if j not in used and distances[i][j] < merge_threshold:
                            to_merge.append(all_contours[j])
                            used.add(j)
                    
                    if len(to_merge) > 1:
                        merged.append(np.concatenate(to_merge))
                    elif i not in used:
                        merged.append(all_contours[i])
                        
                return merged
        
        return all_contours


    def _create_depth_visualization(self, cv_depth, depth_meters):
        """Create visualization of depth image with colored regions."""
        depth_vis = cv2.normalize(cv_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_vis_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
        
        # Color regions based on depth thresholds
        too_close = depth_meters < self.min_distance
        too_far = depth_meters > self.max_distance
        good_range = (depth_meters >= self.min_distance) & (depth_meters <= self.max_distance)
        
        depth_vis_color[too_close] = [0, 0, 255]   # Red for too close
        depth_vis_color[too_far] = [255, 0, 0]     # Blue for too far
        depth_vis_color[good_range] = [0, 255, 0]  # Green for good range
        
        # Add legend - using old-style string formatting
        cv2.putText(depth_vis_color, "Too close < %.1fm" % self.min_distance, 
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(depth_vis_color, "Good range",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(depth_vis_color, "Too far > %.1fm" % self.max_distance,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                        
        return depth_vis_color


    def image_callback(self, color_msg, depth_msg, odom_msg):
        try:
            # Convert images
            cv_color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            debug_img = cv_color.copy()

            # Ensure depth image has same size as color image
            if cv_depth.shape[:2] != cv_color.shape[:2]:
                cv_depth = cv2.resize(cv_depth, (cv_color.shape[1], cv_color.shape[0]))

            # Pre-process depth image and compute gradients once
            depth_float = cv_depth.astype(np.float32)  # Convert to float for better precision
            self.sobelx = cv2.Sobel(depth_float, cv2.CV_32F, 1, 0, ksize=3)
            self.sobely = cv2.Sobel(depth_float, cv2.CV_32F, 0, 1, ksize=3)
            
            # Pre-process color image
            blurred = cv2.GaussianBlur(cv_color, (self.blur_kernel_size, self.blur_kernel_size), 0)
            hsv_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Create depth visualization
            depth_meters = cv_depth * 0.001  # Convert to meters
            depth_vis_color = self._create_depth_visualization(cv_depth, depth_meters)

            # Initialize detection array
            detection_array = ColorDetection3DArray()
            detection_array.header = color_msg.header
            detection_array.odom = odom_msg

            # Process each color
            for color_name, color_params in self.colors.iteritems():
                # Detect contours
                contours = self._detect_color(hsv_image, color_params)
                
                # Draw contours on debug image
                cv2.drawContours(debug_img, contours, -1, color_params['rgb'], 2)

                # Process each contour
                for contour in contours:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        # Calculate centroid
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])

                        # Create detection
                        detection_result = self.create_detection3d_msg(
                            contour, cv_depth, color_name, color_msg.header
                        )

                        if detection_result:
                            detection, is_ground = detection_result
                            detection_array.detections.append(detection)
                            
                            # Enhanced debug visualization
                            confidence = detection.detection.results[0].score
                            cv2.circle(debug_img, (cx, cy), 5, (0, 255, 255), -1)
                            
                            # Main label with type and confidence
                            label = "{0} ({1:.2f})".format(
                                "GROUND" if is_ground else "OBJECT", 
                                confidence
                            )
                            cv2.putText(debug_img, label, (cx-20, cy-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Combine and publish debug image
            h_stack = np.hstack([debug_img, depth_vis_color])
            debug_msg = self.bridge.cv2_to_imgmsg(h_stack, "bgr8")
            debug_msg.header = color_msg.header
            self.debug_image_pub.publish(debug_msg)

            # Publish detections
            if detection_array.detections:
                self.detection3d_pub.publish(detection_array)

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
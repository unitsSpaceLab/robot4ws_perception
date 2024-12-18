#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ColorDetector:
    def __init__(self):
        rospy.init_node('color_detector', anonymous=True)
        
        self.bridge = CvBridge()
        
        self.load_parameters()
        
        if self.debug_visualization:
            self.debug_publisher = rospy.Publisher("/color_detection/debug_image", Image, queue_size=1)
        
        self.image_sub = rospy.Subscriber(
            self.color_topic,
            Image,
            self.image_callback,
            queue_size=1
        )
        
        # (BGR format)
        self.viz_colors = {
            'red': (0, 0, 255),      
            'blue': (255, 0, 0),     
            'green': (0, 255, 0),    
            'yellow': (0, 255, 255), 
            'orange': (0, 165, 255), 
            'purple': (255, 0, 255), 
            'cyan': (255, 255, 0)   
        }

    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # Get topics
        self.color_topic = rospy.get_param('camera_topics/color')
        
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

    def preprocess_image(self, image):
        """Apply preprocessing to the image"""
        # Apply Gaussian blur (is used to reduce noise)
        blurred = cv2.GaussianBlur(image, 
                                  (self.blur_kernel_size, self.blur_kernel_size), 
                                  0)
        
        # Convert to HSV (HSV is more powerful than RGB)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return hsv

    def detect_color(self, hsv_image, color_name):
        """Detect specific color in the image"""
        color_params = self.colors[color_name]
        
        # Create mask for the color
        mask = cv2.inRange(hsv_image, 
                          color_params['lower'], 
                          color_params['upper'])
        
        # Apply morphological operations (erosion followed by dilatio, always to clean the image from noises)
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        # Filter contours by area
        valid_contours = [cnt for cnt in contours 
                         if cv2.contourArea(cnt) > self.min_contour_area]
        
        return valid_contours

    def create_debug_image(self, image, all_detections):
        """Create debug visualization image with all colors"""
        debug_image = image.copy()
        
        # Draw detections for each color
        for color_name, contours in all_detections.items():
            viz_color = self.viz_colors[color_name]
            
            # Draw contours
            cv2.drawContours(debug_image, contours, -1, viz_color, 2)
            
            # Draw color name and area for each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    text = "{}: {:.0f}".format(color_name, area)
                    cv2.putText(debug_image, 
                               text,
                               (cx-20, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.5,
                               viz_color,
                               2)
        
        return debug_image

    def image_callback(self, msg):
        """Process incoming image messages"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Preprocess image
            hsv_image = self.preprocess_image(cv_image)
            
            # Store all detections
            all_detections = {}
            
            # Detect each color
            for color_name in self.colors:
                contours = self.detect_color(hsv_image, color_name)
                all_detections[color_name] = contours
                
                # Log detections
                # if contours:
                #    rospy.loginfo("Detected {} {} objects".format(len(contours), color_name))
            
            # Create and publish debug visualization if enabled
            if self.debug_visualization:
                debug_image = self.create_debug_image(cv_image, all_detections)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                self.debug_publisher.publish(debug_msg)
                    
        except CvBridgeError as e:
            rospy.logerr("CV Bridge error: {}".format(e))
        except Exception as e:
            rospy.logerr("Error processing image: {}".format(e))

    def run(self):
        """Run the node"""
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = ColorDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
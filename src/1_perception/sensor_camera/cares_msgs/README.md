# cares_msgs
Package for common ROS message types used across CARES projects.

## Install
This package has no dependencies or other install requirements beyond ROS 1.0: Note - **NONE** should be added.

## Usage
Add a message/serivce/action message as required for CARES projects and update the README to provide information on the message.
**DO NOT** add source code to this package as it is reserved for ROS message types only!!

## Messages

### InstanceMasks.msg
Instance segmentation message for instance segmentation or semantic segmentation detection methods - e.g. pytorch_object_detection or tf_object_detection.

### StereoCameraInfo.msg
Stereo Camera information for calibrated stereo pairs - contains intrinsic and extrinsic information for the stereo pair.

## Services

### CalibrationService.srv
Service message for the calibration service in https://github.com/UoA-CARES/stereo_calibration

### Detection.srv
Service message for instance or semantic segmentation methods - uses InstanceMasks.msg

### ArucoDetect.srv
Service message for stereo aruco detection 

## Actions

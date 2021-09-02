std_msgs/Header header

# ID for the associated image
uint32 image_id

# Bounding boxes in pixels
sensor_msgs/RegionOfInterest[] boxes

# Integer label IDs for each bounding box
int32[] labels

# Float probability scores of the class_id
float32[] scores

# Instance masks as Images
# 0: Mask [0 - 1]
sensor_msgs/Image[] masks

# Instance masks as Images
# 0: Mask [0 - 1]
# 1: Label
# 2: Instance
sensor_msgs/Image aggregated_mask
int32 id
float64 time
bool isKeyframe

# camToWorld as serialization of sophus sim(3).
# may change with keyframeGraph-updates.
float32[7] camToWorld 

# camera parameter (fx fy cx cy), width, height
# will never change, but required for display.
float32 fx
float32 fy
float32 cx
float32 cy
uint32 height
uint32 width

# RGB Image and Depth Image for visualization with Siasa
sensor_msgs/Image rgb
sensor_msgs/Image depth
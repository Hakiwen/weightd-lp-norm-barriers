import rospy
import tf

from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose

class UnicycleGazeboInterface :
    def __init__(self):
        rospy.init_node("unicycle_gazebo_interface")
        self.pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.sub = rospy.Subscriber("states", Float32MultiArray, self.states_callback, queue_size=1)


    def states_callback(self, states):
        model_state_msg = ModelState()
        model_state_msg.pose = Pose()
        model_state_msg.twist = Twist()

        model_state_msg.pose.position.x = states[0]
        model_state_msg.pose.position.y = states[1]
        model_state_msg.pose.position.z = 0

        mode_state_msg.pose.orientation = tf.createQuaternionMsgFromYaw(states[2])

        model_state_msg.model_name = "unicycle"
        model_state_msg.reference_frame = "world"
        self.pub.publish(model_state_msg)

    def loop(self):
        while not rospy.is_shutdown():
            rospy.spin()
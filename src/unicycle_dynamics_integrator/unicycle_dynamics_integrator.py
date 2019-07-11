import rospy
import tf
import numpy as np

from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import ModelState

class UnicycleDynamicsIntegrator:
    def __init__(self):
        self.x = np.zeros(3)
        self.u = np.zeros(2)
        rospy.init_node("unicycle_dynamics_integrator")
        self.input_sub = rospy.Subscriber("inputs", Float32MultiArray, self.inputs_callback, queue_size=1)
        self.states_pub = rospy.Publisher("states", Float32MultiArray, queue_size=1)
        self.freq = 100.0
        self.period = 1/self.freq
        self.rate = rospy.Rate(self.freq)

    def inputs_callback(self, inputs_msg):
        self.u = inputs_msg.data

    def integrate(self):
        self.x[0] = self.u[0]*np.cos(self.x[2])*self.period + self.x[0]
        self.x[1] = self.u[0]*np.sin(self.x[2])*self.period + self.x[1]
        self.x[2] = self.u[1]*self.period + self.x[2]
        state_msg = Float32MultiArray()
        state_msg.data = self.x
        self.states_pub.publish(state_msg)

    def loop(self):
        while not rospy.is_shutdown():
            self.integrate()
            self.rate.sleep()

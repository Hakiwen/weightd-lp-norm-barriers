import rospy
import osqp
import numpy as np
from math import pow, sqrt, cos, sin, atan2, pi

from std_msgs.msg import Float32MultiArray

global mu, kappa, sigma_1, sigma_2, sigma_3, l, c, p, theta_0, phi_term, mu1, mu2, rho, gamma, large, delta, mu_hat_max, epsilon_1, epsilon_2

mu_hat_max = delta*pow(sigma_3,p)/pow(2*pi, p)

def R_gamma(x_bar):
    x = x_bar[0]
    y = x_bar[1]
    return sqrt(pow(kappa*x,2) + pow(kappa*y + 1,2))

def grad_R_gamma(x_bar):
    divisor = sqrt(pow(kappa*x_bar[0],2) + pow(kappa*x_bar[1] + 1,2))
    x_term = pow(kappa,2)*x_bar[0]/divisor
    y_term = 2*kappa*(kappa*x_bar[1] + 1)/divisor
    return [x_term, y_term]

def theta_gamma(x_bar):
    x = x_bar[0]
    y = x_bar[1]
    return atan2(kappa*y + 1, x)

def grad_theta_gamma(x_bar):
    divisor = pow(kappa*x_bar[0],2) + pow(kappa*x_bar[1] + 1, 2)
    x_term = -kappa*(kappa*x_bar[1] + 1)/divisor
    y_term = pow(kappa,2)*x_bar[0]/divisor
    return [x_term, y_term]

def alpha(X):
    x_bar = X[0:1]
    phi = X[2]
    alpha_terms = np.zeros(3)
    alpha_terms[0] = pow(((R_gamma(x_bar) - c)/sigma_2),p)
    alpha_terms[1] = pow(((theta_gamma(x_bar) - theta_0)/sigma_1),p)
    alpha_terms[2] = mu*pow((abs(kappa)*(phi - phi_term)/sigma_3),p)
    return alpha_terms.sum()

def psi(phi):
    return mu*kappa*pow(phi - phi_term,p - 1)/pow(sigma_3,p)

def omega(x_bar):
    omega_terms = np.zeros(2)
    omega_terms[0] = pow(abs(R_gamma(x_bar) - 1)/sigma_2, p)
    omega_terms[1] = pow(abs(theta_gamma(x_bar) - theta_0)/sigma_1, p)
    return pow(omega_terms.sum(), 1/p)

def grad_omega(x_bar):
    R = R_gamma(x_bar)
    theta = theta_gamma(x_bar)
    scalar_term = pow(pow(1/sigma_2, p)*pow(abs(R - 1), p) + pow(1/sigma_1, p)*abs(theta - theta_0), 1/p - 1)/p
    a = [p*pow(1/sigma_2, p)*pow(abs(R - 1), p - 1), p*pow(1/sigma_1, p)*pow(1/sigma_1, p)*pow(abs(theta - theta_0), p - 1)]
    b = np.concatenate((grad_R_gamma(x_bar), grad_theta_gamma(x_bar)), axis=0)
    return scalar_term*np.matmul(a,b)

def RL_bar(phi):
    R = np.zeros(2,2)
    R[0,0] = cos(phi)
    R[0,1] = -l*sin(phi)
    R[1,0] = sin(phi)
    R[1,1] = l*cos(phi)
    return R

def g(X):
    phi = X[2]
    g_func = np.zeros(3,2)
    g_func[0,0] = cos(phi)
    g_func[0,1] = -l*sin(phi)
    g_func[2,1] = sin(phi)
    g_func[2,2] = -l*cos(phi)
    g_func[3,1] = 0
    g_func[3,2] = 1
    return g_func

def L_g_Upsilon(X):
    x_bar = X[0:1]
    return pow(alpha(X), (1-p)/p)*(pow(-omega(x_bar),p - 1)*grad_omega(x_bar)*RL_bar(X[2]) - psi(X[2])*[0, 1])

def upsilon(X):
    x_bar = X[0:1]
    first_term = abs(kappa)
    second_term = np.zeros(3)
    second_term[0] = mu1*pow((R_gamma(x_bar) - c)/sigma_2,p)
    second_term[1] = mu1*pow((theta_gamma(x_bar) - theta_0)/sigma_1,p)
    second_term[2] = mu2*pow((abs(kappa)*(X[2] - phi_term))/sigma_3,p)
    return first_term - pow(second_term.sum(),1/p)

def grad_upsilon(X):
    scalar_term = -pow(upsilon(X),-1)
    x_bar = X[0:1]
    phi = X[2]
    R = R_gamma(x_bar)
    theta = theta_gamma(x_bar)
    a = [mu1*p*pow(((R - c)/sigma_2),p-1)/sigma_2, mu1*p*pow((theta - theta_0)/sigma_1,p -1)/sigma_1]
    b = np.concatenate((grad_R_gamma(x_bar), grad_theta_gamma(x_bar)),axis=0)
    angle_term = kappa*mu2*pow(kappa*(phi - phi_term)/sigma_3, p - 1)/sigma_3
    return scalar_term*np.concatenate((np.matmul(a,b), angle_term), axis=1)

# r here is actually r~
def qp_contraint_rs(r):
    ups = upsilon(r)
    return -gamma*np.sign(ups)*pow(abs(ups), rho)


def qp_constraint_ls(r):
    return np.matmul(grad_upsilon(r), g(r))

def x2r(X):
    r = np.zeros(3)
    r[0] = X[0] + l*cos(X[2])
    r[1] = X[1] + l*sin(X[2])
    r[2] = X[2]
    return r

def r2x(r):
    X = np.zeros(3)
    X[0] = r[0] - l*cos(r[2])
    X[1] = r[1] - l*sin(r[2])
    X[2] = r[2]
    return X


class UnicycleDynamicsIntegrator:
    def __init__(self):
        self.x = np.zeros(3)
        self.u = np.zeros(2)
        rospy.init_node("unicycle_dynamics_integrator")
        self.input_pub = rospy.Publisher("inputs", Float32MultiArray, queue_size=1)
        self.states_sub = rospy.Subscriber("states", Float32MultiArray, self.states_callback, queue_size=1)
        self.freq = 100
        self.period = 1/self.freq
        self.rate = rospy.Rate(self.freq)
        self.qp = osqp.OSQP()
        self.iteration = 0
        self.DeadLock = 0
        self.stage = 1

    def states_callback(self, states_msg):
        r = x2r(states_msg)
        lie_derivative = L_g_Upsilon(r)
        barrier_level = upsilon(r)
        self.algorithm_1(lie_derivative, barrier_level)

        if self.iteration == 0:
            self.qp.setup(P=np.identity(2), A=qp_constraint_ls(r), q=np.zeros(2), l=qp_contraint_rs(r), u=np.inf, **settings)
        else:
            self.qp.update(q=np.zeros(2), l=qp_contraint_rs(r), u=np.inf)

    def algorithm_1(self, lie_derivative, barrier_level):
        if self.DeadLock == 1:
            if self.stage == 1:
                if barrier_level < 0:
                    mu1 = 1
                    # add callbacks for mu_star, mu_hat_max
                    mu2 = min(mu_star, mu_hat_max)
                else:
                    mu1 = 0
                    mu2 = large
                    self.stage = 2
            else:
                if barrier_level < 0:
                    mu1 = 0
                    mu2 = large
                else:
                    mu1 = 1
                    mu2 = min(mu_star, mu_hat_max)
                    self.stage = 1
        else:
            mu1 = 1
            mu2 = large
            if np.linalg.norm(lie_derivative) < gamma:
                self.DeadLock = 1
            else:
                self.DeadLock = 0


    def loop(self):
        while not rospy.is_shutdown():
            rospy.spin()
            self.integrate()
            self.rate.sleep()
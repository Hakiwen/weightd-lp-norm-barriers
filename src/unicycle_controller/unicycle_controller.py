import rospy
import osqp
import numpy as np
from scipy import sparse
from math import pow, sqrt, cos, sin, atan2, pi

from std_msgs.msg import Float32MultiArray


class UnicycleController:

    def __init__(self):
        self.x = np.zeros(3)
        self.u = np.zeros(2)

        rospy.init_node("unicycle_dynamics_integrator")
        self.input_pub = rospy.Publisher("inputs", Float32MultiArray, queue_size=1)
        self.states_sub = rospy.Subscriber("states", Float32MultiArray, self.states_callback, queue_size=1)

        self.qp = osqp.OSQP()
        self.iteration = 0
        self.DeadLock = 0
        self.stage = 1

        self.kappa = rospy.get_param("~kappa")
        self.sigma_1 = rospy.get_param("~sigma_1")
        self.sigma_2 = rospy.get_param("~sigma_2")
        self.sigma_3 = rospy.get_param("~sigma_3")
        self.p = rospy.get_param("~p")
        self.c = rospy.get_param("~c")
        self.l = rospy.get_param("~l")
        self.theta_0 = rospy.get_param("~theta_0")
        self.phi_term = rospy.get_param("~phi_term")
        self.mu_1 = rospy.get_param("~mu_1")
        self.mu_2 = rospy.get_param("~mu_2")
        self.rho = rospy.get_param("~rho")
        self.gamma = rospy.get_param("~gamma")
        self.large = rospy.get_param("~large")
        self.delta = rospy.get_param("~delta")
        self.epsilon_1 = rospy.get_param("~epsilon_1")
        self.epsilon_2 = rospy.get_param("~epsilon_2")

        self.min_grad, self.min_omega = self.min_norm_grad_omega()
        self.mu_hat_max = self.delta * pow(self.sigma_3, self.p) / pow(2 * pi, self.p)
        self.mu_star = self.min_grad * self.l * pow(self.sigma_3, self.p) * pow(self.min_omega, self.p - 1) / \
                       (pow(abs(self.kappa), self.p) * pow(2 * pi, self.p - 1))
    def R_gamma(self, x_bar):
        x = x_bar[0]
        y = x_bar[1]
        return sqrt(pow(self.kappa*x, 2) + pow(self.kappa*y + 1, 2))

    def grad_R_gamma(self, x_bar):
        divisor = sqrt(pow(self.kappa*x_bar[0], 2) + pow(self.kappa*x_bar[1] + 1, 2))
        x_term = pow(self.kappa, 2)*x_bar[0]/divisor
        y_term = 2*self.kappa*(self.kappa*x_bar[1] + 1)/divisor
        return [x_term, y_term]

    def theta_gamma(self, x_bar):
        x = x_bar[0]
        y = x_bar[1]
        return atan2(self.kappa*y + 1, x)

    def grad_theta_gamma(self, x_bar):
        divisor = pow(self.kappa*x_bar[0],2) + pow(self.kappa*x_bar[1] + 1, 2)
        x_term = -self.kappa*(self.kappa*x_bar[1] + 1)/divisor
        y_term = pow(self.kappa,2)*x_bar[0]/divisor
        return [x_term, y_term]

    def alpha(self, X):
        x_bar = X[0:2]
        phi = X[2]
        alpha_terms = np.zeros(3)
        alpha_terms[0] = self.mu_1 * pow(((self.R_gamma(x_bar) - self.c) / self.sigma_2), self.p)
        alpha_terms[1] = self.mu_1 * pow(((self.theta_gamma(x_bar) - self.theta_0) / self.sigma_1), self.p)
        alpha_terms[2] = self.mu_2 * pow((abs(self.kappa) * (phi - self.phi_term) / self.sigma_3), self.p)
        return alpha_terms.sum()

    def psi(self, phi):
        return self.mu_2 * self.kappa * pow(phi - self.phi_term, self.p - 1) / pow(self.sigma_3, self.p)

    def omega(self, x_bar):
        omega_terms = np.zeros(2)
        omega_terms[0] = pow(abs(self.R_gamma(x_bar) - 1) / self.sigma_2, self.p)
        omega_terms[1] = pow(abs(self.theta_gamma(x_bar) - self.theta_0) / self.sigma_1, self.p)
        return pow(omega_terms.sum(), 1 / self.p)

    def grad_omega(self, x_bar):
        R = self.R_gamma(x_bar)
        theta = self.theta_gamma(x_bar)
        scalar_term = pow(pow(1 / self.sigma_2, self.p) * pow(abs(R - 1), self.p)
                          + pow(1 / self.sigma_1, self.p) * abs(theta - self.theta_0), 1 / self.p - 1) / self.p
        a = [self.p * pow(1 / self.sigma_2, self.p) * pow(abs(R - 1), self.p - 1),
             self.p * pow(1 / self.sigma_1, self.p) * pow(1 / self.sigma_1,
             self.p) * pow(abs(theta - self.theta_0), self.p - 1)]
        b = np.vstack((self.grad_R_gamma(x_bar), self.grad_theta_gamma(x_bar)))
        return scalar_term*np.matmul(a, b)

    def RL_bar(self, phi):
        R = np.zeros((2,2))
        R[0,0] = cos(phi)
        R[0,1] = -self.l * sin(phi)
        R[1,0] = sin(phi)
        R[1,1] = self.l * cos(phi)
        return R

    def g(self, X):
        phi = X[2]
        g_func = np.zeros((3,2))
        g_func[0,0] = cos(phi)
        g_func[0,1] = -self.l * sin(phi)
        g_func[1,0] = sin(phi)
        g_func[1,1] = -self.l * cos(phi)
        g_func[2,0] = 0
        g_func[2,1] = 1
        return g_func

    def L_g_Upsilon(self, X):
        x_bar = X[0:2]
        return pow(self.alpha(X), (1 - self.p) / self.p) \
            * (pow(-self.omega(x_bar), self.p - 1) * np.matmul(self.grad_omega(x_bar), self.RL_bar(X[2]))
               - self.psi(X[2]) * np.asarray([0, 1]))

    def upsilon(self, X):
        x_bar = X[0:2]
        first_term = abs(self.kappa)
        second_term = np.zeros(3)
        second_term[0] = self.mu_1 * pow((self.R_gamma(x_bar) - self.c) / self.sigma_2, self.p)
        second_term[1] = self.mu_1 * pow((self.theta_gamma(x_bar) - self.theta_0) / self.sigma_1, self.p)
        second_term[2] = self.mu_2 * pow((abs(self.kappa) * (X[2] - self.phi_term)) / self.sigma_3, self.p)
        return first_term - pow(second_term.sum(), 1 / self.p)

    def grad_upsilon(self, X):
        scalar_term = -pow(self.upsilon(X), -1)
        x_bar = X[0:2]
        phi = X[2]
        R = self.R_gamma(x_bar)
        theta = self.theta_gamma(x_bar)
        a = [self.mu_1 * self.p * pow(((R - self.c) / self.sigma_2), self.p - 1) / self.sigma_2,
             self.mu_1 * self.p * pow((theta - self.theta_0) / self.sigma_1, self.p - 1) / self.sigma_1]
        b = np.vstack((self.grad_R_gamma(x_bar), self.grad_theta_gamma(x_bar)))
        angle_term = self.kappa * self.mu_2 * pow(self.kappa * (phi - self.phi_term) / self.sigma_3, self.p - 1) \
                     / self.sigma_3
        return scalar_term*np.hstack((np.matmul(a,b), angle_term))


    def checkDonut(self, x, y, epsilon):
        R = self.R_gamma([x, y])
        theta = self.theta_gamma([x,y])

        alpha = abs(R - self.c) / self.sigma_2
        beta = abs(theta - self.theta_0) / self.sigma_1

        Hg = abs(self.kappa) - np.linalg.norm([alpha, beta], self.p)
        H_eps = np.linalg.norm([alpha, beta], self.p) - abs(self.kappa) - epsilon

        if Hg >= 0 and H_eps >= 0:
            d = 0
        else:
            d = 1
        return d

    # may not be epsilon_1
    def min_norm_grad_omega(self):
        theta_k = 2*pi/3
        epsilon = self.epsilon_1
        t_mu = (self.kappa + epsilon)/self.kappa
        x_Lp = np.linspace(-t_mu*self.sigma_1, t_mu*self.sigma_1, 100)
        y_Lp = np.linspace(-t_mu*self.sigma_2, t_mu*self.sigma_2, 100)
        x, y = np.meshgrid(x_Lp, y_Lp, sparse=False, indexing='ij')
        R_k = 1/abs(self.kappa)
        alpha_b = np.sign(self.kappa)

        Bx = np.multiply((R_k + y), np.cos(alpha_b*pi/2 + theta_k/2*x/self.sigma_1))
        By = np.multiply((R_k + y), np.sin(alpha_b*pi/2 + theta_k/2*x/self.sigma_1)) - alpha_b*R_k

        Dx = []
        Dy = []
        m = 1
        for i in range(Bx.shape[0]):
            for j in range(Bx.shape[0]):
                donut = self.checkDonut(Bx[i,j], By[i,j], epsilon)
                if donut == 1:
                    Dx.append(Bx[i,j])
                    Dy.append(By[i,j])
                    m += 1

        D = np.hstack((Dx, Dy))
        G = np.zeros((len(Dx), 2))
        Norm_Grad = []
        omega_vals = []
        for i in range(len(Dx)):
            x_new_B = self.kappa*Dx[i]
            y_new_B = self.kappa*Dy[i]
            R_B = sqrt(pow(x_new_B, 2) + np.power(y_new_B, 2))
            theta_B = atan2(y_new_B, x_new_B)
            Lp_weight_B_p_1 = pow(pow(R_B - self.c, self.p) / pow(self.sigma_2, self.p)
                                  + pow(theta_B - self.theta_0, self.p) / pow(self.sigma_1, self.p), 1 / self.p - 1)

            V_v = Lp_weight_B_p_1 * self.p * np.asarray([[pow(R_B - self.c, self.p - 1) / pow(self.sigma_2, self.p)],
                                            [-pow(theta_B - self.theta_0, self.p - 1) / pow(self.sigma_1, self.p)]])
            Q_inv = [[cos(theta_B), sin(theta_B)], [(1/R_B)*sin(theta_B), -(1/R_B)*cos(theta_B)]]
            G[i, :] = np.matmul(V_v.T, Q_inv)
            Norm_Grad.append(np.linalg.norm(G[i, :]))
            omega_vals.append(self.omega([x_new_B, y_new_B]))

        return min(Norm_Grad), min(omega_vals)

    # r here is actually r~
    def qp_contraint_rs(self, r):
        ups = self.upsilon(r)
        return -self.gamma * np.sign(ups) * pow(abs(ups), self.rho)

    def qp_constraint_ls(self, r):
        # print self.grad_upsilon(r)
        return np.matmul(self.grad_upsilon(r), self.g(r))

    def x2r(self, X):
        r = np.zeros(3)
        r[0] = X[0] + self.l * cos(X[2])
        r[1] = X[1] + self.l * sin(X[2])
        r[2] = X[2]
        return r

    def r2x(self, r):
        X = np.zeros(3)
        X[0] = r[0] - self.l * cos(r[2])
        X[1] = r[1] - self.l * sin(r[2])
        X[2] = r[2]
        return X

    def states_callback(self, states_msg):
        r = self.x2r(states_msg.data)
        print r
        lie_derivative = self.L_g_Upsilon(r)
        barrier_level = self.upsilon(r)
        self.algorithm_1(lie_derivative, barrier_level)

        # osqp needs np.ndarray for l, cannot construt from a scalar
        P_qp = sparse.csc_matrix(np.identity(2))
        A_qp = sparse.csc_matrix(np.vstack((self.qp_constraint_ls(r), [0, 0])))
        l_qp = np.asarray([self.qp_contraint_rs(r),  -1])
        u_qp = np.asarray([np.inf, np.inf])
        # A_qp = sparse.csc_matrix(self.qp_constraint_ls(r))
        # l_qp = np.asarray([self.qp_contraint_rs(r)])
        # u_qp = np.inf
        # print A_qp
        # print l_qp
        if self.iteration == 0:
            self.qp.setup(P=P_qp,
                          A=A_qp,
                          q=np.zeros(2),
                          l=l_qp,
                          u=u_qp,
                          verbose=False)
        else:
            self.qp.update(q=np.zeros(2),
                           l=l_qp,
                           u=u_qp)
        self.iteration += 1
        qp_results = self.qp.solve()

        print type(qp_results.x)
        print l_qp
        print qp_results.x
        print self.qp_constraint_ls(r)
        print np.matmul(self.qp_constraint_ls(r), qp_results.x)
        print barrier_level
        inputs_msg = Float32MultiArray()
        inputs_msg.data = qp_results.x
        self.input_pub.publish(inputs_msg)

    def algorithm_1(self, lie_derivative, barrier_level):
        # print lie_derivative
        # print barrier_level
        # print self.DeadLock
        if self.DeadLock == 1:
            if self.stage == 1:
                if barrier_level < 0:
                    self.mu_1 = 1
                    # add callbacks for mu_star, mu_hat_max
                    self.mu_2 = min(self.mu_star, self.mu_hat_max)
                else:
                    self.mu_1 = 0
                    self.mu_2 = self.large
                    self.stage = 2
            else:
                if barrier_level < 0:
                    self.mu_1 = 0
                    self.mu_2 = self.large
                else:
                    self.mu_1 = 1
                    self.mu_2 = min(self.mu_star, self.mu_hat_max)
                    self.stage = 1
        else:
            self.mu_1 = 1
            self.mu_2 = self.large
            if np.linalg.norm(lie_derivative) < self.gamma:
                self.DeadLock = 1
            else:
                self.DeadLock = 0

    def loop(self):
        while not rospy.is_shutdown():
            rospy.spin()

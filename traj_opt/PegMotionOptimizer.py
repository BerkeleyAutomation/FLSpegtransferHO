import numpy as np
import FLSpegtransferHO.motion.dvrkVariables as dvrkVar
from FLSpegtransferHO.motion.dvrkKinematics import dvrkKinematics
import time
import osqp
from scipy import sparse
from threading import Thread

def add_pen_var(q, A, l, u, penalty):
    pen_var = len(q)
    q.append(penalty)
    A.append([len(l), pen_var, 1])
    l.append(0.0)
    u.append(np.inf)
    return pen_var

def penalize_next_constraint(q, A, l, u, penalty):
    """Creates penalty variables and adds them to the next constraint.

    This must be called before starting the constraint to be penalized since
    it creates two constraints in the process of setting up the slack
    variables.
    """
    pen_lo = add_pen_var(q, A, l, u, penalty)
    pen_hi = add_pen_var(q, A, l, u, penalty)
    A.append([len(l), pen_lo, 1.0])
    A.append([len(l), pen_hi, -1.0])



    """

2-waypoint
     ____________
  q1/            \q2
   |              |
   |              |
   q0             q3


1-waypoint (lift)
     __________q2
  q1/            
   | dir=(0,0,1)
   |             
   q0            


1-waypoint (drop)
    q0___________
                 \q1
                  | always dir=(0,0,-1)
                  |
                  q2

"""

class Spline:
    def __init__(self, dt, q0, v0, q1, v1):
        self.c0 = (2*q0 - 2*q1 +   dt*v0 + dt*v1) / (dt*dt*dt);
        self.c1 = (3*q1 - 3*q0 - 2*dt*v0 - dt*v1) / (dt*dt);
        self.c2 = v0;
        self.c3 = q0;

    def pos(self, t):
        t2 = t*t
        return t*t2*self.c0 + t2*self.c1 + t*self.c2 + self.c3

    def vel(self, t):
        return 3*t*t*self.c0 + 2*t*self.c1 + self.c2

    def acc(self, t):
        return 6*t*self.c0 + 2*self.c1

class Point:
    def __init__(self, x, dim, t_step, H):
        self.x = x
        self.t_step = t_step
        self.H = H
        self.dim = dim
        self.v0idx = (H+1)*dim
        self.a0idx = (H+1)*2*dim

    def pos(self, t):
        return self.x[t*self.dim:(t+1)*self.dim]
    def vel(self, t):
        return self.x[self.v0idx + t*dim:self.v0idx + (t+1)*dim]
    def acc(self, t):
        return self.x[self.a0idx + t*dim:self.a0idx + (t+1)*dim]

    def set_pos(self, t, q):
        self.x[t*self.dim:(t+1)*self.dim] = q
    def set_vel(self, t, v):
        self.x[self.v0idx + t*dim:self.v0idx + (t+1)*dim] = v
    def set_acc(self, t, a):
        self.x[self.a0idx + t*dim:self.a0idx + (t+1)*dim] = a

def interpolate(x_prev, H_prev, t_prev, H_next, t_next, n_slack=0):
    dim = 6
    rate = H_prev * t_prev / (H_next * t_next)
    t_spline = t_prev / rate

    vp0i = (H_prev+1)*dim
    
    vn0i = (H_next+1)*dim
    an0i = (H_next+1)*dim*2

    x_next = np.zeros(((H_next+1)*3-1) * dim + n_slack)
    
    for t in range(H_next):
        t0 = t * H_prev // H_next
        s = (t * H_prev % H_next) * t_spline / H_next
        q0 = x_prev[        t0   *dim:       (t0+1)*dim]
        q1 = x_prev[       (t0+1)*dim:       (t0+2)*dim]
        v0 = x_prev[vp0i +  t0   *dim:vp0i + (t0+1)*dim]
        v1 = x_prev[vp0i + (t0+1)*dim:vp0i + (t0+2)*dim]
        spline = Spline(t_spline, q0, v0*rate, q1, v1*rate)
        
        x_next[       t*dim:       (t+1)*dim] = spline.pos(s)
        x_next[vn0i + t*dim:vn0i + (t+1)*dim] = spline.vel(s)
        x_next[an0i + t*dim:an0i + (t+1)*dim] = spline.acc(s)

    # copy last
    x_next[       H_next*dim:       (H_next+1)*dim] = x_prev[       H_prev*dim:       (H_prev+1)*dim]
    x_next[vn0i + H_next*dim:vn0i + (H_next+1)*dim] = x_prev[vp0i + H_prev*dim:vp0i + (H_prev+1)*dim] * rate
    
    # with open('interpolate.gp', 'w') as f:
    #     print("$PREV << EOD", file=f)
    #     for t in range(H_prev):
    #         q = x_prev[t*dim:(t+1)*dim]
    #         v = x_prev[(H_prev+t+1)*dim:(H_prev+t+2)*dim]
    #         a = x_prev[(H_prev*2+t+2)*dim:(H_prev*2+t+3)*dim]
    #         print(*q, *v, *a, file=f)
    #     print("EOD", file=f)
    #     print("$NEXT << EOD", file=f)
    #     for t in range(H_next): # +1 needed for full q and v
    #         q = x_next[t*dim:(t+1)*dim]
    #         v = x_next[(H_next+t+1)*dim:(H_next+t+2)*dim]
    #         a = x_next[(H_next*2+t+2)*dim:(H_next*2+t+3)*dim]
    #         print(*q, *v, *a, file=f)
    #     print("EOD", file=f)
    #     print("plot '$PREV' u 0:1 w l, '' u 0:7 w l, '' u 0:13 w l, '$NEXT' u 0:1 w l, '' u 0:7 w l, '' u 0:13 w l", file=f)
    # import sys
    # sys.exit(0)
        
    return x_next

class _PegMotionQP:
    def __init__(self, H, t_step):
        self.robot = dvrkKinematics()
        self.max_vel = [1.0, 1.0, 1.0, 8.0, 8.0, 8.0]
        self.max_acc = [1.0, 1.0, 1.0, 8.0, 8.0, 8.0]
        self.t_step = t_step
        self.H = H
        self.dim = 6

        # H+1 q, H+1 v, H a
        self._n_vars = (H+1) * self.dim * 3 - self.dim
        self._v0 = (H+1) * self.dim
        self._a0 = (H+1) * self.dim * 2
        self.init()

    def init(self):
        self._Pval = []
        self._Prow = []
        self._Pcol = []
        self._Aval = []
        self._Arow = []
        self._Acol = []
        self._q = [0.0] * self._n_vars
        self._l = []
        self._u = []

        # minimize sum of squared accelerations
        for t in range(self.H):
            for j in range(self.dim):
                self._Prow.append(self._aidx(t, j))
                self._Pcol.append(self._aidx(t, j))
                self._Pval.append(1.0)

        # dynamics
        for t in range(self.H):
            for j in range(self.dim):
                # q_{t+1} ==             q_t + v_t * t_step + a_t * t_step**2/2
                # 0       == - q_{t+1} + q_t + v_t * t_step + a_t * t_step**2/2
                self._constrain(self._qidx(t+1, j), -1.0)
                self._constrain(self._qidx(t, j), 1.0)
                self._constrain(self._vidx(t, j), self.t_step)
                self._constrain(self._aidx(t, j), self.t_step**2/2)
                self._bound(0.0, 0.0)

                self._constrain(self._vidx(t+1, j), -1.0)
                self._constrain(self._vidx(t, j), 1.0)
                self._constrain(self._aidx(t, j), self.t_step)
                self._bound(0.0, 0.0)

        # velocity limits
        for t in range(0, self.H+1):
            for j in range(self.dim):
                self._constrain(self._vidx(t, j), 1.0)
                self._bound(-self.max_vel[j], self.max_vel[j])

        # acceleration limit
        for t in range(0, self.H):
            for j in range(self.dim):
                self._constrain(self._aidx(t, j), 1.0)
                self._bound(-self.max_acc[j], self.max_acc[j])

    def constrain_q(self, t, q):
        """Adds a constraint for time t to a at configuration"""
        for j in range(self.dim):
            self._constrain(self._qidx(t, j), 1.0)
            self._bound(q[j], q[j])

    def constrain_v(self, t, v):
        """Add a constraint for time t to be at velocity"""
        for j in range(self.dim):
            self._constrain(self._vidx(t, j), 1.0)
            self._bound(v[j], v[j])

    def constrain_pos(self, t, q_prev, pos, tolerance, penalty):
        scale = 10.0
        J_prev = self.robot.jacobian(q_prev)[0]
        p_prev = self.robot.fk(q_prev)[:3,3]
        for i in [0, 1]: # x and y
            self._penalize_constraint(penalty)
            c = pos[i] - p_prev[i] # center line
            for j in range(self.dim):
                self._constrain(self._qidx(t, j), J_prev[i][j] * scale)
                c += J_prev[i][j] * q_prev[j]
            self._bound((c - tolerance) * scale, (c + tolerance) * scale)

    def _constrain(self, i, v):
        """Adds a variable to the current linear constraint"""
        assert isinstance(v, float)
        self._Arow.append(len(self._l))
        self._Acol.append(i)
        self._Aval.append(v)

    def _bound(self, lo, up):
        """Sets the bounds on the current constraint and starts the next"""
        self._l.append(lo)
        self._u.append(up)

    def _add_penalty_var(self, penalty):
        """Creates a slack variable for a penalty"""
        pen_var = len(self._q)
        self._q.append(penalty)
        self._constrain(pen_var, 1.0)
        self._bound(0.0, np.inf)
        return pen_var

    def _penalize_constraint(self, penalty):
        """Adds two slack variables to the current contraint.

        This must be called before calling _constrain the first time for the
        constraint.
        """
        # order is important here.  we create two penalty variables
        # which both add to both the variables and the constraints
        pen_lo = self._add_penalty_var(penalty)
        pen_hi = self._add_penalty_var(penalty)
        # We then add these variables to the current constraint.
        self._constrain(pen_lo,  1.0)
        self._constrain(pen_hi, -1.0)

    def _qidx(self, t, j=0):
        """Returns the index of the configuration at time t, joint j"""
        return t*self.dim + j
    
    def _vidx(self, t, j=0):
        """Returns the index of the velocity at time t, joint j"""
        return self._v0 + t*self.dim + j
    
    def _aidx(self, t, j=0):
        """Returns the index of the acceleration at time t, joint j"""
        return self._a0 + t*self.dim + j

    def solve(self, x_prev=None):
        solver = osqp.OSQP()
        n = len(self._q)
        m = len(self._l)
        P = sparse.csc_matrix((self._Pval, (self._Prow, self._Pcol)), shape=(n,n))
        A = sparse.csc_matrix((self._Aval, (self._Arow, self._Acol)), shape=(m,n))
        q = np.array(self._q)
        l = np.array(self._l)
        u = np.array(self._u)
        solver.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, adaptive_rho=True, adaptive_rho_interval=100, verbose=False, rho=0.1)
        if x_prev is not None:
            solver.warm_start(x=x_prev)

        r = solver.solve()
        if r.info.status != "solved":
            return None

        return r.x

    # def resolve(self, x_prev=None):
    #     n = len(self._q)
    #     m = len(self._l)
    #     #P = sparse.csc_matrix((self._Pval, (self._Prow, self._Pcol)), shape=(n,n))
    #     A = sparse.csc_matrix((self._Aval, (self._Arow, self._Acol)), shape=(m,n))
    #     assert(A.has_sorted_indices)
    #     q = np.array(self._q)
    #     l = np.array(self._l)
    #     u = np.array(self._u)
    #     #self.solver.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)
    #     self.solver.update(Ax=A.data, Ax_idx=A.indices, q=q, l=l, u=u)
    #     if x_prev is not None:
    #         self.solver.warm_start(x=x_prev)

    #     r = self.solver.solve()
    #     if r.info.status != "solved":
    #         return None

    #     return r.x
        


class PegMotionOptimizerV2b:
    def __init__(self, max_vel, max_acc, t_step):
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.t_step = t_step

    def compute_lift(self, q0, q1, H=25):
        dim = 6
        robot = dvrkKinematics()
        qp = _PegMotionQP(H=H, t_step=self.t_step)
        qp.constrain_q(0, q0)
        qp.constrain_v(0, np.zeros(6))
        qp.constrain_q(H, q1)
        x = qp.solve()

        p0 = robot.fk(q0)[:3,3]
        p1 = robot.fk(q1)[:3,3]

        for h in range(H-1, 3, -1):
            # for iter_no in range(3):
            qp = _PegMotionQP(H=h, t_step=self.t_step)
            qp.constrain_q(0, q0)
            qp.constrain_v(0, np.zeros(6))
            qp.constrain_q(h, q1)
            for t in range(1,h):
                q_prev = x[qp._qidx(t):qp._qidx(t+1)]
                p_prev = (p0*(h-t) + p1*t)/h
                qp.constrain_pos(t, q_prev, p_prev, tolerance=1e-3/4, penalty=1e+3)

            x_prev = None # interpolate(x, H, self.t_step, h, self.t_step, n_slack=(h-1)*4)
            x_next = qp.solve(x_prev) #x if iter_no > 0 else None)
            if x_next is None:
                break
            x,H = x_next,h
        
        return x,H

    def compute_drop(self, q2, q3, H=25):
        robot = dvrkKinematics()
        qp = _PegMotionQP(H=H, t_step=self.t_step)
        qp.constrain_q(0, q2)
        qp.constrain_q(H, q3)
        qp.constrain_v(H, np.zeros(6))
        x = qp.solve()

        p2 = robot.fk(q2)[:3,3]
        p3 = robot.fk(q3)[:3,3]

        for h in range(H-1, 3, -1):
            qp = _PegMotionQP(H=h, t_step=self.t_step)
            qp.constrain_q(0, q2)
            qp.constrain_q(h, q3)
            qp.constrain_v(h, np.zeros(6))
            for t in range(1,h):
                q_prev = x[qp._qidx(t):qp._qidx(t+1)]
                p_prev = (p2*(h-t) + p3*t)/h
                qp.constrain_pos(t, q_prev, p_prev, tolerance=1e-3/4, penalty=1e+3)

            x_next = qp.solve() # x if iter_no > 0 else None)
            if x_next is None:
                break
            x,H = x_next,h
        
        return x, H

    def optimize_motion(self, q0, q1, q2, q3):
        dim = 6

        # Hard code: 0.75 second upper limit of lift/drop time
        x01,t01 = self.compute_lift(q0, q1, H=int(0.75/self.t_step))
        x23,t23 = self.compute_drop(q2, q3, H=int(0.75/self.t_step))
        
        v01i = (t01+1)*dim + t01*dim
        v23i = (t23+1)*dim

        # Hard code: 1.5 second upper limit of time between lift and drop
        t12 = int(1.5/self.t_step)
        for h in range(t12, 3, -1):
            qp = _PegMotionQP(h, t_step=self.t_step)
            qp.constrain_q(0, q1)
            qp.constrain_v(0, x01[v01i:v01i+dim])
            qp.constrain_q(h, q2)
            qp.constrain_v(h, x23[v23i:v23i+dim])
            x12_next = qp.solve()
            if x12_next is None:
                break
            x12,t12 = x12_next,h

        H = t01 + t12 + t23
        x = np.zeros((H+1)*3*dim - dim)
        
        x[0:(t01+1)*dim] = x01[0:(t01+1)*dim]
        x[(t01+1)*dim:(t01+t12)*dim] = x12[dim:t12*dim]
        x[(t01+t12)*dim:(H+1)*dim] = x23[0:(t23+1)*dim]

        print(t01, t12, t23)
        return x,H
        
    
class PegMotionOptimizerV2a:
    def __init__(self, max_vel, max_acc, t_step=0.01):
        self.robot = dvrkKinematics()
        self.solver = None
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.t_step = t_step

    # def optimize_lift(self, q0, q1, v0, v1, H):
    #     dim = 6
    #     P,A,q,l,u = [],[],[],[],[]
        
    #     x_prev = np.zeros(n)
    #     for t in range(H):
    #         for j in range(dim):
    #             x_prev.append(

    #     # boundary conditions
    #     for t,qi,vi in [[0, q0, v0], [H, q1, v1]]:
    #         for j in range(dim):
    #             A.append([len(l), q_var(t, j), 1.0])
    #             l.append(qi[j])
    #             u.append(qi[i])
    #         if vi is not None:
    #             for j in range(dim):
    #                 A.append([len(l), v_var(t, j), 1.0])
    #                 l.append(vi[j])
    #                 u.append(vi[j])
    #     # dynamics
    #     for t in range(H):
    #         for j in range(dim):
    #             A.append([len(l), q_var(t+1, j), -1.0])
    #             A.append([len(l), q_var(t, j), 1.0])
    #             A.append([len(l), v_var(t, j), t_step])
    #             A.append([len(l), a_var(t, j), t_step*t_step/2])
    #             l.append(0.0)
    #             u.append(0.0)
    #             A.append([len(l), v_var(t+1, j), -1.0])
    #             A.append([len(l), v_var(t, j), 1.0])
    #             A.append([len(l), a_var(t, j), t_step])
    #             l.append(0.0)
    #             u.append(0.0)

    #     # limits
    #     for t in range(0,H+1):
    #         for j in range(dim):
    #             A.append([len(l), v_var(t, j), 1])
    #             l.append(-max_vel[j])
    #             u.append( max_vel[j])
                
    #     # limits
    #     for t in range(0,H):
    #         for j in range(dim):
    #             A.append([len(l), a_var(t, j), 1])
    #             l.append(-max_acc[j])
    #             u.append( max_acc[j])

    #     # objective: sum of squared velocities
    #     for t in range(H):
    #         for j in range(dim):
    #             P.append([v_var(t,j), v_var(t,j), 1.0])

    #     for t in range(1,H):
    #         q_prev = x_prev[q_var(t):q_var(t+1)]
    #         J_prev = self.robot.jacobian(q_prev)
    #         p_prev = self.robot.fk(q_prev)
    #         for i in [0,1]: # x and y
    #             penalize_next_constraint(q, A, l, u, penalty)
    #             bc = center_pos[i] - p_prev[i]
    #             for j in range(dim):
    #                 A.append([len(l), q_var(t, j), J_prev[i][j] * peg_scale])
    #                 bc += J_prev[i][j] * q_prev[j]
    #             l.append((bc - peg_gap) * peg_scale)
    #             u.append((bc + peg_gap) * peg_scale)
            

    def optimize_motion(self, q0, q1, q2, q3, max_vel, max_acc, t_step=0.01, horizon=50, print_out=False, visualize=False):
        dim = 6
        H = horizon

        v0index = dim*(H+1)
        a0index = v0index*2
        n = v0index*3-dim

        # 0 = time at q0
        t1 = 19   # time at q1
        t2 = H-18 # time at q2
        # H = time at q3

        # x = [ q_{t=0,j=0:6}, q_{t=1,j=0:6}, ... ,
        #       v_{t=0,j=0:6}, v_{t=1,j=0:6}, ... , 
        #       a_{t=0,j=0:6}, a_{t=1,j=0:6}, ... ,
        #       penalty variables ]
        def q_var(t, j=0):  # q_var(2, 5) = index of configuration at time = 2, for joint 5
            return t*dim + j
        def v_var(t, j=0):
            return v0index + t*dim + j
        def a_var(t, j=0):
            return a0index + t*dim + j

        # initialize x by interpolation
        x_prev = np.zeros(n)
        for t in range(0, t1):
            p = t/t1
            x_prev[q_var(t):q_var(t+1)] = q0*(1-p) + q1*p

        for t in range(t1, t2+1):
            p = (t-t1)/(t2-t1)
            x_prev[q_var(t):q_var(t+1)] = q1*(1-p) * q2*p

        for t in range(t2, H+1):
            p = (t - t2)/(H+1-t2)
            x_prev[q_var(t):q_var(t+1)] = q2*(1-p) * q2*p
                
        for t in range(H):
            x_prev[v_var(t):v_var(t+1)] = (x_prev[q_var(t+1):q_var(t+2)] - x_prev[q_var(t):q_var(t+1)])/t_step
                
        for t in range(H):
            x_prev[a_var(t):a_var(t+1)] = (x_prev[v_var(t+1):v_var(t+2)] - x_prev[v_var(t):v_var(t+1)])/t_step


        penalty = 1e5
        for iter_no in range(5):
            # set up a QP in the form:
            #
            # min 1/2 x^T P x + q^T x
            # s.t. l <= A x <= u
            P,A,q,l,u = [],[],[0.0]*n,[],[]
        
            # hit each waypoint
            for t,qi in [[0, q0], [H, q3], [t1, q1], [t2, q2]]:
                for j in range(dim):
                    # q_{t,j} = qi[j]
                    A.append([len(l), q_var(t, j), 1.0])
                    l.append(qi[j])
                    u.append(qi[j])

            # start and stop at v=0
            for t in [0, H]:
                for j in range(dim):
                    A.append([len(l), v_var(t, j), 1])
                    l.append(0.0)
                    u.append(0.0)
                                    
            # dynamics
            # q_{t+1} = q_{t} + v_{t} * t_step + a_{t} * t_step**2 / 2
            # v_{t+1} = v_{t} + a_{t} * t_step
            for t in range(H):
                for j in range(dim):
                    A.append([len(l), q_var(t+1, j), -1.0])
                    A.append([len(l), q_var(t, j), 1.0])
                    A.append([len(l), v_var(t, j), t_step])
                    A.append([len(l), a_var(t, j), t_step*t_step/2])
                    l.append(0.0)
                    u.append(0.0)
                    A.append([len(l), v_var(t+1, j), -1.0])
                    A.append([len(l), v_var(t, j), 1.0])
                    A.append([len(l), a_var(t, j), t_step])
                    l.append(0.0)
                    u.append(0.0)

            # velocity and acceleration limits
            if True:
                for t in range(0,H+1):
                    for j in range(dim):
                        A.append([len(l), v_var(t, j), 1.0])
                        l.append(-max_vel[j])
                        u.append(max_vel[j])
            if True:
                for t in range(0,H):
                    for j in range(dim):
                        A.append([len(l), a_var(t, j), 1.0])
                        l.append(-max_acc[j])
                        u.append( max_acc[j])

            # Objective: minimize sum-of-squared velocities
            #           [ 1 0    ]
            # 1/2 v^T *   0 1    ] * v
            #                ... ]
            for t in range(H):
                for j in range(dim):
                    P.append([v_var(t,j), v_var(t,j), 1.0])
        

            # Use Jacobian to enforce that velocity in x and y is zero.
            # (0,0,+) = J * v
            #
            # p1 = p0 + J (q1 - q0)
            # p1 = p0 + J q1 - J q0
            #
            # lb             <= p1 <= ub
            #             lb <= p0 + J q1 - J q0 <= ub
            # lb + J q0 - p0 <=      J q1        <= ub + J q0 - p0
            peg_scale = 1e1
            peg_gap = 1e-3

            if True:
                peg_pos = [ 0.145, -0.018 ] # TODO: get from data
                for t in range(1, t1):
                    J = self.robot.jacobian(x_prev[q_var(t):q_var(t+1)])[0]
                    p0 = self.robot.fk(x_prev[q_var(t):q_var(t+1)])[:3,3]
                    for i in [0,1]: # x and y
                        penalize_next_constraint(q, A, l, u, penalty)                        
                        bc = peg_pos[i] - p0[i] # bounds center
                        for j in range(dim):
                            A.append([len(l), q_var(t, j), J[i][j] * peg_scale])
                            bc += J[i][j] * x_prev[q_var(t, j)]
                        l.append((bc - peg_gap) * peg_scale)
                        u.append((bc + peg_gap) * peg_scale)

                peg_pos = [ 0.110, -0.032 ]
                for t in range(t2+1,H):
                    J = self.robot.jacobian(x_prev[q_var(t):q_var(t+1)])[0]
                    p0 = self.robot.fk(x_prev[q_var(t):q_var(t+1)])[:3,3]
                    print(t,p0)
                    for i in [0,1]:
                        penalize_next_constraint(q, A, l, u, penalty)                        
                        bc = peg_pos[i] - p0[i] # bounds center
                        for j in range(dim):
                            A.append([len(l), q_var(t, j), J[i][j] * peg_scale])
                            bc += J[i][j] * x_prev[q_var(t, j)]
                        l.append((bc - peg_gap) * peg_scale)
                        u.append((bc + peg_gap) * peg_scale)
                    

            n = len(q)
            m = len(l)
            A = sparse.csc_matrix(([v for r,c,v in A], ([r for r,c,v in A], [c for r,c,v in A])), shape=(m,n))
            P = sparse.csc_matrix(([v for r,c,v in P], ([r for r,c,v in P], [c for r,c,v in P])), shape=(n,n))

            q = np.array(q)
            self.solver = osqp.OSQP()
            self.solver.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-5, eps_rel=1e-5, max_iter=100000)
            if x_prev.shape[0] == n:
                self.solver.warm_start(x=x_prev)
            # TODO: update on subsequent iterations
            # self.solver.update(Ax=A, l=l, u=u)
                
            r = self.solver.solve()
            if r.info.status != "solved":
                return None
            x_prev = r.x
            assert(x_prev.shape[0] == n)
            
        #print(r.x)
        return x_prev


if __name__ == "__main__":
    # dvrkVar.v_max,
    max_vel = [1.0, 1.0, 1.0, 8.0, 8.0, 8.0]     # max velocity (rad/s) or (m/s)
    # dvrkVar.a_max
    max_acc = [1.0, 1.0, 1.0, 8.0, 8.0, 8.0]     # max acceleration (rad/s^2) or (m/s^2)
    t_step = 0.1
    opt = PegMotionOptimizerV2b(max_vel, max_acc, t_step)
    
    waypoints = np.load("traj_opt/script/ref_waypoints.npy")

    wp = waypoints[0]
    robot = dvrkKinematics()
    #print("wp0:", robot.fk(wp[2])[:3,3])
    #print("wp1:", robot.fk(wp[3])[:3,3])
    #H = 72
    start = time.time()
    x,H = opt.optimize_motion(*wp) #, max_vel=max_vel, max_acc=max_acc, t_step=0.03, horizon=H)
    elapsed = time.time() - start
    print("Solve time:", elapsed)
    if x is not None:
        dim = 6
        with open("peg_motion_optimizer_v2.gp", "w") as f:
            print("$DATA << EOD", file=f)
            for t in range(H+1):
                q = x[t*dim:(t+1)*dim]
                p = robot.fk(q)
                t = p[:3,3]
                print(*q, *t, file=f)
            print("EOD", file=f)
            print("$WP << EOD", file=f)
            for q in wp:
                p = robot.fk(q)
                t = p[:3,3]
                print(*q, *t, file=f)
            print("EOD", file=f)
            # print("plot for [i=1:6] '$DATA' u 0:i w l", file=f)
            print("set parametric", file=f)
            print("set xlabel 'x'", file=f)
            print("set ylabel 'y'", file=f)
            print("set zlabel 'z'", file=f)
            print("set xrange [ 0.128-0.020: 0.128+0.020]", file=f)
            print("set yrange [-0.025-0.020:-0.025+0.020]", file=f)
            print("splot '$DATA' u 7:8:9 w lp lw 3 not, '$WP' u 7:8:9 w p ps 5 t 'waypoints'", file=f)


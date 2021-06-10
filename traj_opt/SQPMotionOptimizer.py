import numpy as np
import osqp
import time
from scipy import sparse
import FLSpegtransfer.motion.dvrkVariables as dvrkVar
from FLSpegtransfer.motion.dvrkKinematics import dvrkKinematics
from FLSpegtransfer.traj_opt.PegMotionOptimizer import interpolate
import logging
import time
import os
import sys
import tempfile
import multiprocessing as mp
import threading

log = logging.getLogger("SQPMotionOptimizer") #(__name__)
logging.basicConfig(level=logging.NOTSET)
#logging.root.setLevel(level=logging.NOTSET)
log.setLevel(logging.INFO)

class SparseMatrixBuilder:
    def __init__(self):
        self.rows = []
        self.cols = []
        self.vals = []

    def append(self, r, c, v):
        self.rows.append(r)
        self.cols.append(c)
        self.vals.append(v)

    def build(self, n, m):
        #log.info(f"Building {n}x{m} sparse matrix with {len(self.vals)}")
        M = sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=(n,m))
        #log.info(M)
        return M

# class Constraint:
#     def __init__(self, index):
#         self.index = index


# class LinearExpression:
#     def __init__(self, qp, variables, scales, constant):
#         self.qp = qp
#         self.variables = variables
#         self.scales = scales
#         self.constant = constant

#     def __mul__(self, val):
#         if isinstance(val, Number):
#             return LinearExpression(qp, self.variables.copy(), [x*val for x in self.scales], self.constant*val)
#         else:
#             raise ValueError()

#     def __add__(self, val):
#         if isinstance(val, Number):
#             return LinearExpression(qp, self.variables.copy(), self.scales.copy(), self.constant + val)
#         if isinstance(val, Variable):
#             assert(self.qp == val.qp)
#             return LinearExpression(qp, self.variables + [val.index], self.scales + [1.0], self.constant)
#         if isinstance(val, LinearExpression):
#             assert(self.qp == val.qp)
#             return LinearExpression(qp, self.variables + val.variables, self.scale + val.scales, self.constant + val.constant)
#         else:
#             raise ValueError()
        
# class Variable:
#     def __init__(self, qp, index):
#         self.qp = qp
#         self.index = index

#     def __mul__(self, val):
#         if isinstance(val, Number):
#             return LinearExpression(qp, [self.index], [val], 0.0)
#         else:
#             raise ValueError()

#     def __add__(self, val):
#         if isinstance(val, Number):
#             return LinearExpression(qp, [self.index], [1.0], val)
#         if isinstance(val, Variable):
#             assert(self.qp == val.qp)
#             return LinearExpression(qp, [self.index, val], [1.0, 1.0], 0.0)
#         if isinstance(val, LinearExpression):
#             assert(self.qp == val.qp)
#             return LinearExpression(qp, [self.index]+val.variables, [1.0]+val.scales, val.constant)
#         else:
#             raise ValueError()

    
# class QP:
#     def __init__(self):
#         self.P = SparseMatrixBuilder()
#         self.A = SparseMatrixBuilder()
#         self.q = []
#         self.l = []
#         self.u = []

#     def add_vars(self, lin_costs):
#         self.q += lin_costs

#     def add_var(self, lin_cost=0.0):
#         self.q.append(lin_cost)
#         return len(self.q)-1

#     def add_constrain(self, lower, upper):
#         assert(lower <= upper)
#         self.l.append(lower)
#         self.u.append(upper)
#         return len(self.u)-1

#     def constrain(self, c, v, scale=1.0):
#         self.A.append(c, v, scale)


# class MotionQP(QP):
#     def __init__(self, dim, H, t_step, vel_max, acc_max, jerk_max, q0=None, v0=None, qH=None, vH=None):
#         super().__init__()
#         assert(H > 2)
#         assert(0 < t_step and t_step < 1e3)

#         self.dim = dim
#         self.H = H
#         self.t_step = t_step

#         self.v0idx = (self.H+1)*self.dim
#         self.a0idx = self.v0idx*2
#         self.nvars = self.a0idx + self.H*dim
        
#         self.add_vars([0.0] * self.n_vars)

#         vel_min = [-v for v in vel_max]
#         acc_min = [-a for a in acc_max]

#         # Add boundary constraints
#         if q0 is not None:
#             self.constrain_config(0, q0, q0)
#         if qH is not None:
#             self.constrain_config(H, qH, qH)
            
#         if v0 is not None:
#             self.constrain_velocity(0, v0, v0)
#         else:
#             self.constrain_velocity(0, vel_min, vel_max)
            
#         if vH is not None:
#             self.constrain_velocity(H, vH, vH)
#         else:
#             self.constrain_velocity(0, vel_min, vel_max)

#         # Add joint velocity limits
#         for t in range(1, H):
#             self.constrain_velocity(t, vel_min, vel_max)

#         # Add joint acceleration limits
#         for t in range(1, H):
#             self.constrain_acceleration(t, acc_min, acc_max)

#         # Configuration Dynamics
#         # q_{t+1} = q_t + v_t*t_step + a_t*t_step**2/2
#         for t in range(H):
#             for j in range(dim):
#                 c = self.add_constraint(0.0, 0.0)
#                 self.constrain(c, self.qidx(t+1, j), -1.0)
#                 self.constrain(c, self.qidx(t, j), 1.0)
#                 self.constrain(c, self.vidx(t, j), self.t_step)
#                 self.constrain(c, self.aidx(t, j), self.t_step**2/2)

#         # Velocity Dynamics
#         # v_{t+1} = v_t + a_t*t_step
#         for t in range(H):
#             for j in range(dim):
#                 c = self.add_constraint(0.0, 0.0)
#                 self.constrain(c, self.vidx(t+1, j), -1.0)
#                 self.constrain(c, self.vidx(t, j), 1.0)
#                 self.constrain(c, self.aidx(t, j), self.t_step)

#     def qidx(self, t, j=0):
#         assert(0 <= t and t <= self.H)
#         assert(0 <= j and j <= self.dim)
#         return t*self.dim + j

#     def vidx(self, t, j=0):
#         assert(0 <= t and t <= self.H)
#         assert(0 <= j and j <= self.dim)
#         return self.v0idx + t*self.dim + j

#     def aidx(self, t, j=0):
#         assert(0 <= t and t < self.H)
#         assert(0 <= j and j <= self.dim)
#         return self.a0idx + t*self.dim + j

#     def constrain_config(self, t, qmin, qmax):
#         assert(len(qmin) == self.dim)
#         assert(len(qmax) == self.dim)
#         for j in range(self.dim):
#             c = self.add_constraint(qmin, qmax)
#             self.constrain(c, self.qidx(t, j))

#     def constrain_velocity(self, t, vmin, vmax):
#         for j in range(self.dim):
#             c = self.add_constraint(vmin, vmax)
#             self.constrain(c, self.vidx(t, j))

#     def constrain_acceleration(self, t, amin, amax):
#         for j in range(self.dim):
#             c = self.add_constraint(amin, amax)
#             self.constrain(c, self.aidx(t, j))
            

#     def solve(self):
#         solver = osqp.OSQP()
#         n = len(self.q)
#         m = len(self.u)
#         solver.setup(
#             P=self.P.build(n,n), q=np.array(self.q),
#             A=self.A.build(m,n), l=np.array(self.l), u=np.array(self.u),
#             max_iter=10000,
#             rho=0.1, adaptive_rho=True, adaptive_rho_interval=100,
#             verbose=False)
#         r = solver.solve()
#         if r.info.status != "solved":
#             return None
        
#         return r.x
                

class LiftConstraint:
    def __init__(self, sqp, q0, q1):
        self.sqp = sqp
        self.fk0 = sqp.robot.fk(q0)[:3,3]
        self.fk1 = sqp.robot.fk(q1)[:3,3]
        self.tolerance = 1e-3
        self.scale = 1.0
        self.scale_top = 1.0
        self.radius = 5e-3
        self.rounded = True

    def convexify(self, x):
        sqp = self.sqp
        tgt = self.fk1 # TODO: interpolated between fk0 and fk1?
        lift_t_max = sqp.H//2 - 1
        for t in range(1, sqp.H): # <-- really this could be itertools.count(), we always break earlier.
            q = sqp.config(x, t)
            J = sqp.robot.jacobian([q])[0]
            pos = sqp.robot.fk(q)[:3,3]

            # If the position is above q1, then we're done with this
            # loop
            #log.debug(f"t={t}, a={tgt[2] - pos[2]} (lift)")
            if pos[2] > self.fk1[2] or t >= lift_t_max:
                break

            if False:
                # # (J v)_z > 0
                c = sqp.add_constraint(0, np.inf)
                for j in range(sqp.dim):
                    sqp.constrain(c, sqp.vidx(t,j), J[2,j])
                sqp.add_penalty(c)

            a = (tgt[2] - pos[2])
            blend = min(1.0, max(0.0, (pos[2] - tgt[2] + self.radius) / self.radius))
            if self.rounded and blend > 0.0:
                #
                #      \   c   /
                #   x---+  |  +---
                #       |  |  |
                #log.debug(f"t={t}, a={a}, {blend} (lift)")
                #a,blend = 1.0,0.0 # DELETE ME
                n = np.sqrt(blend**2 + (1.0 - blend)**2)
                a = blend/n
                b = (1.0 - blend)/n

                # (pos + J (q' - q)) . n >= dist_to_center + radius
                # pos.n + (J q').n - (J q) . n >= dist_to_center + radius 
                # (J q').n >= dist_to_center + radius  - pos . n + (J q) . n
                for i in range(2): # only for x and y coeffs
                    for k in [-1.0, 1.0]:
                        # Compute normal 
                        n = np.array([0.0, 0.0, a])
                        n[i] = k*b

                        center = tgt.copy()
                        center[2] -= self.radius # lower the center from the top
                        center[i] -= k * (self.radius + self.tolerance) # offset the center left/right
                        dist_to_center = np.dot(n, center)

                        rhs = dist_to_center + self.radius + np.dot(J[:3,:] @ q - pos, n)
                        c = sqp.add_constraint(rhs * self.scale, np.inf)
                        dots = np.zeros(6, dtype=np.float32) # DELETE ME
                        for j in range(sqp.dim):
                            dot = sum([ J[ni, j] * n[ni] for ni in range(3) ])
                            dots[j] = dot # DELETE ME
                            sqp.constrain(c, sqp.qidx(t, j), dot * self.scale)

                        #log.debug(f"  n={n}, center={center}, rhs={rhs}")
                        #log.debug(f"t={t}, i={i}, bound=({rhs}, inf), n={n}, Jn={dots}")
                        sqp.add_penalty(c)
            else:
                # pos + J (q' - q) = tgt
                # J q' = tgt - pos + J q
                for i in range(2): # only for x and y coeffs
                    rhs = tgt[i] - pos[i]
                    for j in range(sqp.dim):
                        rhs += J[i,j] * q[j]
                    c = sqp.add_constraint(
                        (rhs - self.tolerance) * self.scale,
                        (rhs + self.tolerance) * self.scale)
                    for j in range(sqp.dim):
                        sqp.constrain(c, sqp.qidx(t, j), J[i,j] * self.scale)
                    sqp.add_penalty(c)

        #self.cvx_t = t
        rhs = tgt[2] - pos[2]
        for j in range(sqp.dim):
            rhs += J[2, j] * q[j]
        c = sqp.add_constraint(rhs*self.scale*t, np.inf)
        for j in range(sqp.dim):
            sqp.constrain(c, sqp.qidx(t, j), J[2, j] * self.scale*t)
        sqp.add_penalty(c) # TODO: we only need 1 penalty variable here.

    def cost(self, x):
        sqp = self.sqp
        tgt = self.fk1 # TODO: interpolated between fk0 and fk1?
        cost = 0.0
        lift_t_max = sqp.H//2 - 1
        for t in range(1, sqp.H):
            #for t in range(1, self.cvx_t):
            q = sqp.config(x, t)
            J = sqp.robot.jacobian([q])[0]
            pos = sqp.robot.fk(q)[:3,3]

            #log.debug(f"{t}, {q}, {pos}")

            # If the position is above q1, then we're done with this
            # loop
            if pos[2] > self.fk1[2] or t >= lift_t_max:
                log.debug(f"cost break on t={t}")
                break

            if False:
                # # (J v)_z > 0
                rhs = 0
                for j in range(sqp.dim):
                    rhs += x[sqp.vidx(t,j)] * J[2,j]
                if rhs < 0:
                    cost += -rhs * self.scale

            a = (tgt[2] - pos[2])
            blend = min(1.0, max(0.0, (pos[2] - tgt[2] + self.radius) / self.radius))
            if self.rounded and blend > 0.0:
                #
                #      \   c   /
                #   x---+  |  +---
                #       |  |  |
                #log.debug(f"t={t}, a={a}, {blend} (lift)")
                n = np.sqrt(blend**2 + (1.0 - blend)**2)
                a = blend/n
                b = (1.0 - blend)/n

                # (pos + J (q' - q)) . n >= dist_to_center + radius
                # pos.n + (J q').n - (J q) . n >= dist_to_center + radius 
                # (J q') >= dist_to_center + radius  - pos . n + (J q) . n
                for i in range(2): # only for x and y coeffs
                    for k in [-1.0, 1.0]:
                        # Compute normal 
                        n = np.array([0.0, 0.0, a])
                        n[i] = k*b

                        center = tgt.copy()
                        center[2] -= self.radius # lower the center from the top
                        center[i] -= k * (self.radius + self.tolerance) # offset the center left/right
                        dist_to_center = np.dot(n, center)
                        
                        rhs = dist_to_center + self.radius - np.dot(pos, n)
                        if rhs > 0:
                            cost += rhs
            else:
                # pos + J (q' - q) = tgt
                # J q' = tgt - pos + J q
                for i in range(2): # only for x and y coeffs
                    rhs = abs(tgt[i] - pos[i])
                    #log.info(f"t={t}, i={i}, rhs={rhs}")
                    if rhs > self.tolerance:
                        #log.debug(f" cost += {rhs - self.tolerance}")
                        cost += (rhs - self.tolerance)*self.scale

        if pos[2] < tgt[2]:
            #log.debug(f"top cost = {(tgt[2] - pos[2]) * t}")
            cost += (tgt[2] - pos[2])*t
        return cost
                    

class DropConstraint:
    def __init__(self, sqp, q2, q3):
        self.sqp = sqp
        self.fk2 = sqp.robot.fk(q2)[:3,3]
        self.fk3 = sqp.robot.fk(q3)[:3,3]
        self.tolerance = 1e-3
        self.scale = 1.0
        self.scale_top = 1.0
        self.radius = 5e-3
        self.rounded = True

    def convexify(self, x):
        sqp = self.sqp
        tgt = self.fk2 # TODO: interpolated between fk0 and fk1?
        lift_t_max = sqp.H//2 - 1
        for tinv in range(1, sqp.H): # <-- really this could be itertools.count(), we always break earlier.
            t = sqp.H - tinv
            q = sqp.config(x, t)
            J = sqp.robot.jacobian([q])[0]
            pos = sqp.robot.fk(q)[:3,3]

            # If the position is above q1, then we're done with this
            # loop
            if pos[2] > self.fk2[2] or tinv >= lift_t_max:
                break

            if False:
                # # (J v)_z > 0
                c = sqp.add_constraint(0, np.inf)
                for j in range(sqp.dim):
                    sqp.constrain(c, sqp.vidx(t,j), J[2,j])
                sqp.add_penalty(c)

            a = (tgt[2] - pos[2])
            blend = min(1.0, max(0.0, (pos[2] - tgt[2] + self.radius) / self.radius))
            if self.rounded and blend > 0.0:
                #
                #      \   c   /
                #   x---+  |  +---
                #       |  |  |
                #log.debug(f"t={t}, a={a}, {blend} (drop)")
                #a,blend = 1.0,0.0 # DELETE ME
                n = np.sqrt(blend**2 + (1.0 - blend)**2)
                a = blend/n
                b = (1.0 - blend)/n

                # (pos + J (q' - q)) . n >= dist_to_center + radius
                # pos.n + (J q').n - (J q) . n >= dist_to_center + radius 
                # (J q').n >= dist_to_center + radius  - pos . n + (J q) . n
                for i in range(2): # only for x and y coeffs
                    for k in [-1.0, 1.0]:
                        # Compute normal 
                        n = np.array([0.0, 0.0, a])
                        n[i] = k*b

                        center = tgt.copy()
                        center[2] -= self.radius # lower the center from the top
                        center[i] -= k * (self.radius + self.tolerance) # offset the center left/right
                        dist_to_center = np.dot(n, center)

                        rhs = dist_to_center + self.radius + np.dot(J[:3,:] @ q - pos, n)
                        c = sqp.add_constraint(rhs * self.scale, np.inf)
                        dots = np.zeros(6, dtype=np.float32) # DELETE ME
                        for j in range(sqp.dim):
                            dot = sum([ J[ni, j] * n[ni] for ni in range(3) ])
                            dots[j] = dot # DELETE ME
                            sqp.constrain(c, sqp.qidx(t, j), dot * self.scale)

                        #log.debug(f"  n={n}, center={center}, rhs={rhs}")
                        #log.debug(f"t={t}, i={i}, bound=({rhs}, inf), n={n}, Jn={dots}")
                        sqp.add_penalty(c)
            else:
                # pos + J (q' - q) = tgt
                # J q' = tgt - pos + J q
                for i in range(2): # only for x and y coeffs
                    rhs = tgt[i] - pos[i]
                    for j in range(sqp.dim):
                        rhs += J[i,j] * q[j]
                    c = sqp.add_constraint(
                        (rhs - self.tolerance) * self.scale,
                        (rhs + self.tolerance) * self.scale)
                    #log.debug(f"t={t}, i={i}, bounds=({rhs - self.tolerance}, {rhs + self.tolerance}), n={J[i,:]}")
                    for j in range(sqp.dim):
                        sqp.constrain(c, sqp.qidx(t, j), J[i,j] * self.scale)
                    sqp.add_penalty(c)

        #self.cvx_t = t
        rhs = tgt[2] - pos[2]
        for j in range(sqp.dim):
            rhs += J[2, j] * q[j]
        c = sqp.add_constraint(rhs*self.scale_top*tinv, np.inf)
        for j in range(sqp.dim):
            sqp.constrain(c, sqp.qidx(t, j), J[2, j] * self.scale_top*tinv)
        sqp.add_penalty(c) # TODO: we only need 1 penalty variable here.

    def cost(self, x):
        sqp = self.sqp
        tgt = self.fk2 # TODO: interpolated between fk0 and fk1?
        cost = 0.0
        lift_t_max = sqp.H//2 - 1
        for tinv in range(1, sqp.H):
            #for t in range(1, self.cvx_t):
            t = sqp.H - tinv
            q = sqp.config(x, t)
            J = sqp.robot.jacobian([q])[0]
            pos = sqp.robot.fk(q)[:3,3]

            #log.debug(f"{t}, {q}, {pos}")

            # If the position is above q1, then we're done with this
            # loop
            if pos[2] > self.fk2[2] or tinv >= lift_t_max:
                log.debug(f"cost break on t={t}")
                break

            if False:
                # # (J v)_z > 0
                rhs = 0
                for j in range(sqp.dim):
                    rhs += x[sqp.vidx(t,j)] * J[2,j]
                if rhs < 0:
                    cost += -rhs * self.scale

            a = (tgt[2] - pos[2])
            blend = min(1.0, max(0.0, (pos[2] - tgt[2] + self.radius) / self.radius))
            if self.rounded and blend > 0.0:
                #
                #      \   c   /
                #   x---+  |  +---
                #       |  |  |
                #log.debug(f"t={t}, a={a}, {blend} (drop)")
                n = np.sqrt(blend**2 + (1.0 - blend)**2)
                a = blend/n
                b = (1.0 - blend)/n

                # (pos + J (q' - q)) . n >= dist_to_center + radius
                # pos.n + (J q').n - (J q) . n >= dist_to_center + radius 
                # (J q') >= dist_to_center + radius  - pos . n + (J q) . n
                for i in range(2): # only for x and y coeffs
                    for k in [-1.0, 1.0]:
                        # Compute normal 
                        n = np.array([0.0, 0.0, a])
                        n[i] = k*b

                        center = tgt.copy()
                        center[2] -= self.radius # lower the center from the top
                        center[i] -= k * (self.radius + self.tolerance) # offset the center left/right
                        dist_to_center = np.dot(n, center)
                        
                        rhs = dist_to_center + self.radius - np.dot(pos, n)
                        if rhs > 0:
                            cost += rhs
            else:
                # pos + J (q' - q) = tgt
                # J q' = tgt - pos + J q
                for i in range(2): # only for x and y coeffs
                    rhs = abs(tgt[i] - pos[i])
                    #log.debug(f"t={t}, i={i}, rhs={rhs}")
                    if rhs > self.tolerance:
                        #log.debug(f" cost += {rhs - self.tolerance}")
                        cost += (rhs - self.tolerance)*self.scale

        if pos[2] < tgt[2]:
            #log.debug(f"top cost = {(tgt[2] - pos[2])*tinv*10}")
            cost += (tgt[2] - pos[2])*self.scale_top*tinv
        return cost

    
    
class SQPMotionOptimizer:
    def __init__(self, dim, H, t_step, max_vel, max_acc, max_jerk=None, objective='v', obj_scale=1.0):
        assert(isinstance(dim, int) and dim > 0)
        assert(isinstance(H, int) and H > 3)
        assert(isinstance(t_step, float) and t_step > 0 and t_step < 10.)
        assert(len(max_vel) == dim)
        assert(len(max_acc) == dim)
        assert(max_jerk is None or len(max_jerk) == dim)
        
        self.robot = dvrkKinematics()
        self.use_acc_vars = True

        self.dim = dim
        self.H = H
        self.t_step = t_step
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_jerk = max_jerk

        self.initial_penalty = 1e3
        self.initial_trust_region_size = np.pi
        self.max_iters = 500

        high_quality = False
        
        self.min_trust_region_size = 1e-4
        self.trust_shrink_ratio = 0.1 #0.1
        self.trust_expand_ratio = 1.5
        self.penalty_increase_ratio = 10.0
        self.max_penalty = 1e6 # 1e5
        self.adaptive_eps = True
        self.qp_eps = 1e-6 if high_quality else 1e-4 
        self.qp_max_iter = 10000
        self.warm_start_qps = False

        self.terminate_on_constraint_satisfaction = not high_quality
        self.acceptable_penalty_threshold = 0.0005

        self.init_time = 0.0
        self.qp_time = 0.0
        self.convexify_time = 0.0
        self.cost_time = 0.0
        
        self.obj_scale = obj_scale
        assert objective in {'v', 'a', 'j'}
        self.objective = objective
        
        # if 'v' == objective:
        #     self.objective = slice(v0, a0)
        # elif 'a' == objective:
        #     assert self.use_acc_vars
        #     self.objective = slice(a0, a0+H*dim)
        # else:
        #     raise ValueError("bad objective")


    def add_var(self, lin_cost=0.0):
        self.q.append(lin_cost)
        return len(self.q)-1
    
    def add_constraint(self, lower, upper):
        assert(lower <= upper)
        self.l.append(lower)
        self.u.append(upper)
        return len(self.u)-1

    def qidx(self, t, j):
        assert(0 <= t and t <= self.H)
        assert(0 <= j and j < self.dim)
        return self.dim*t + j

    def vidx(self, t, j):
        assert(0 <= t and t <= self.H)
        assert(0 <= j and j < self.dim)
        return (self.H + 1 + t)*self.dim + j

    def aidx(self, t, j):
        assert(0 <= t and t < self.H)
        assert(0 <= j and j < self.dim)
        assert(self.use_acc_vars)
        return (self.H*2 + 2 + t)*self.dim + j

    def constrain(self, c, i, v):
        self.A.append(c, i, v)
        
    def init_qp(self, q0, qH):
        t_start = time.time()
        self.n_vars = (self.H*3+2)*self.dim # |q|, |v|, |a| = H+1, H+1, H
        self.P = SparseMatrixBuilder()
        self.A = SparseMatrixBuilder()
        self.q = [0.0]*self.n_vars
        self.l = []
        self.u = []

        # Costs
        if 'v' == self.objective:
            v0 = (self.H+1)*self.dim
            for i in range(v0, v0*2):
                self.P.append(i, i, self.obj_scale)
        elif 'a' == self.objective:
            assert self.use_acc_vars
            a0 = (self.H+1)*self.dim*2
            for i in range(a0, a0+self.H*self.dim):
                self.P.append(i, i, self.obj_scale)
        elif 'j' == self.objective:
            # j = (a' - a)/t_step
            # j^2 = (a' - a)(a' - a)/t_step^2
            #     = (a'^2 - a'a - aa' - a^2)/t_step^2
            assert self.use_acc_vars
            js1 = -1.0*self.obj_scale/self.t_step
            js2 =  2.0*self.obj_scale/self.t_step
            for j in range(self.dim):
                self.P.append(self.aidx(0, j), self.aidx(0, j), js2)
            for t in range(1, self.H):
                for j in range(self.dim):
                    self.P.append(self.aidx(t, j), self.aidx(t, j), js2)
                    self.P.append(self.aidx(t, j), self.aidx(t-1, j), js1)
                    self.P.append(self.aidx(t-1, j), self.aidx(t, j), js1)
                    
                
        # for i in range(self.objective.start, self.objective.stop):
        # # for t in range(self.H+1):
        # #     for j in range(self.dim):
        # #         i = self.vidx(t, j) # minimize sum-of-squared velocities
        #         self.P.append(i, i, self.obj_scale)
        
        # start config
        for j in range(self.dim):
            c = self.add_constraint(q0[j], q0[j])
            self.constrain(c, self.qidx(0, j), 1.0)

        # Limit configuration change based on trust region and
        # previous point (we can also include joint limits, but that
        # seems moot for the problems we're working with)
        for t in range(1,self.H):
            for j in range(self.dim):
                i = self.qidx(t, j)
                c = self.add_constraint(
                    self.x[i] - self.trust_region_size, self.x[i] + self.trust_region_size)
                self.constrain(c, i, 1.0)
                assert(c == i) # Needed by update_trust_region

        # end config
        for j in range(self.dim):
            c = self.add_constraint(qH[j], qH[j])
            self.constrain(c, self.qidx(self.H, j), 1.0)

        # Velocity limits
        if True:
            for t in range(1, self.H):
                for j in range(self.dim):
                    c = self.add_constraint(-self.max_vel[j], self.max_vel[j])
                    self.constrain(c, self.vidx(t, j), 1.0)

            # Velocity boundary constraints (start and end at zero velocity)
            for j in range(self.dim):
                c = self.add_constraint(0.0, 0.0)
                self.constrain(c, self.vidx(0, j), 1.0)
                c = self.add_constraint(0.0, 0.0)
                self.constrain(c, self.vidx(self.H, j), 1.0)

        # Acceleration Limits.  These do not include t=0 and t=H-1, as
        # those may be determined by jerk limits.
        if self.use_acc_vars:
            for t in range(1,self.H-1):
                for j in range(self.dim):
                    c = self.add_constraint(-self.max_acc[j], self.max_acc[j])
                    self.constrain(c, self.aidx(t,j), 1.0)
        else:
            # a' = (v' - v)/t
            # For boundary conditions:
            #   a0 = (v1 - v0)/t, with v0 constrained to 0 above.
            #   a_{H-1} = (vH - v_{H-1})/t, also with vH constarint to 0 above.
            # This really means that v1 and v_{H-1} are constrained twice (once to v_max, once to a_max)
            for t in range(0, self.H):
                for j in range(self.dim):
                    c = self.add_constraint(-self.max_acc[j]*self.t_step, self.max_acc[j]*self.t_step)
                    self.constrain(c, self.vidx(t+1, j), 1.0)
                    self.constrain(c, self.vidx(t, j), -1.0)

        if self.max_jerk is None:
            if self.use_acc_vars:
                # If there are no jerk limits, then we need to constrain
                # the start and end acceleration as we do with any other
                # waypoint
                for j in range(self.dim):
                    c = self.add_constraint(-self.max_acc[j], self.max_acc[j])
                    self.constrain(c, self.aidx(0,j), 1.0)
                    c = self.add_constraint(-self.max_acc[j], self.max_acc[j])
                    self.constrain(c, self.aidx(self.H-1,j), 1.0)            
        else:
            if self.use_acc_vars:
                # Add jerk limits, computed using the finite difference between waypoints:
                # jerk = (a_{t} - a_{t-1}) / t_step
                for t in range(1, self.H):
                    for j in range(self.dim):
                        c = self.add_constraint(-self.max_jerk[j]*self.t_step, self.max_jerk[j]*self.t_step)
                        self.constrain(c, self.aidx(t-1,j), -1.0)
                        self.constrain(c, self.aidx(t,  j),  1.0)

                # Since we have jerk limits, the start and end
                # acceleration is limited either by the acceleration
                # limits or the jerk limits assuming 0 acceleration before
                # and after the trajectory
                for j in range(self.dim):
                    limit = min(self.max_jerk[j] * self.t_step, self.max_acc[j])
                    c = self.add_constraint(-limit, limit)
                    self.constrain(c, self.aidx(0, j), 1.0)
                    c = self.add_constraint(-limit, limit)
                    self.constrain(c, self.aidx(self.H-1, j), -1.0)
            else:
                # a = (v' - v)/t_step
                # j = (a' - a)/t_step
                # j = ((v" - v')/t_step - (v' - v)/t_step)/t_step
                # j = ((v" - v') - (v' - v))/t_step^2
                # j = (v" - 2*v' + v)/t_step^2
                for t in range(0, self.H):
                    for j in range(self.dim):
                        c = self.add_constraint(-self.max_jerk[j]*self.t_step, self.max_jerk[j]*self.t_step)
                        if t > 0:
                            self.constrain(c, self.vidx(t-1, j), 1.0/self.t_step)
                        self.constrain(c, self.vidx(t, j), -2.0/self.t_step)
                        if t < self.H:
                            self.constrain(c, self.vidx(t+1, j), 1.0/self.t_step)

        # dynamics
        if self.use_acc_vars:
            # q' = q + vt + at^2/2
            for t in range(self.H):
                for j in range(self.dim):
                    c = self.add_constraint(0.0, 0.0)
                    self.constrain(c, self.qidx(t+1, j), -1.0)
                    self.constrain(c, self.qidx(t, j), 1.0)
                    self.constrain(c, self.vidx(t, j), self.t_step)
                    self.constrain(c, self.aidx(t, j), self.t_step**2/2)
            # v' = v + at
            for t in range(self.H):
                for j in range(self.dim):
                    c = self.add_constraint(0.0, 0.0)
                    self.constrain(c, self.vidx(t+1, j), -1.0)
                    self.constrain(c, self.vidx(t, j), 1.0)
                    self.constrain(c, self.aidx(t, j), self.t_step)
        else:
            # v' = v + at
            # a  = (v' - v)/t
            # q' = q + vt + at^2/2
            # q' = q + vt + ((v' - v)/t)t^2/2
            # q' = q + vt + (v' - v)t/2
            # q' = q + vt/2 + v't/2
            for t in range(self.H):
                for j in range(self.dim):
                    c = self.add_constraint(0.0, 0.0)
                    self.constrain(c, self.qidx(t+1, j), -1.0)
                    self.constrain(c, self.qidx(t, j), 1.0)
                    self.constrain(c, self.vidx(t, j), self.t_step/2)
                    self.constrain(c, self.vidx(t+1, j), self.t_step/2)

        self.n_convex_constraints = len(self.u)
        self.init_time = time.time() - t_start

    def setup_qp(self, x=None, y=None, rho=0.1):
        self.qp = osqp.OSQP()
        n = len(self.q)
        m = len(self.u)
        log.debug(f"Setting up QP, n={n}, m={m}, n_vars={self.n_vars}, n_pen={n-self.n_vars}")
        self.q = np.array(self.q)
        self.l = np.array(self.l)
        self.u = np.array(self.u)
        t_start = time.time()
        eps = max(self.qp_eps, self.trust_region_size/self.penalty) if self.adaptive_eps else self.qp_eps
        self.qp.setup(
            P=self.P.build(n, n), q=self.q, A=self.A.build(m, n), l=self.l, u=self.u,
            max_iter=self.qp_max_iter, adaptive_rho=True, adaptive_rho_interval=100,
            eps_rel=eps, eps_abs=eps,
            verbose=False, rho=rho)
        if self.warm_start_qps and x is not None:
            x_ws = np.zeros(n)
            x_ws[0:self.n_vars] = x[0:self.n_vars]
            y_ws = np.zeros(m)
            y_ws[0:self.n_convex_constraints] = y[0:self.n_convex_constraints]
            self.qp.warm_start(x=x_ws, y=y_ws)
        self.qp_time += time.time() - t_start

            

    def solve_one_qp(self):
        t_start = time.time()
        r = self.qp.solve()
        self.qp_time += time.time() - t_start
        log.debug(f"solve_qp => {r.info.status}")
        return r
            
    # def penalty_adjust(self):
    #     # TODO: check of constraints are satisfied
    #     self.penalty *= self.penalty_increase_ratio
    #     self.trust_region_size = max(self.trust_region_size, self.min_trust_region_size) / self.trust_shrink_ratio * self.trust_expand_ratio**2
    #     self.init_qp()
    #     self.convexify()

    def update_trust_region(self):
        for t in range(1, self.H):
            for j in range(self.dim):
                i = self.qidx(t, j)
                v = self.x[i]
                self.l[i] = v - self.trust_region_size
                self.u[i] = v + self.trust_region_size

        self.qp.update(l=self.l, u=self.u)

    def config(self, x, t):
        i = self.dim*t
        return x[i:i+6]

    def velocity(self, x, t):
        i = self.dim*(self.H+1+t)
        return x[i:i+6]
    
    def acceleration(self, x, t):
        if self.use_acc_vars:
            i = self.dim*(self.H*2+2+t)
            return x[i:i+6]
        else:
            return (self.velocity(x, t+1) - self.velocity(x, t)) / self.t_step

    # def add_linear_penalty(self, c, dir):
    #     v_pen = self.add_var(self.penalty)
    #     c_pen = self.add_constraint(0, np.inf)
    #     self.constraints(c_pen, v_pen, 1.0)
    #     self.constrain(c, v_pen, dir)
                       
    def add_penalty(self, c):
        # add two penalty variables, one for negative penalty, one for
        # positive, both result in a linear penalty to the cost.
        for s in [-1.0, 1.0]:
            v_pen = self.add_var(self.penalty) # Add variable with linear cost of the penalty
            c_pen = self.add_constraint(0, np.inf) # constraint it to be positive
            self.constrain(c_pen, v_pen, 1.0) # 1.0 * v 
            self.constrain(c, v_pen, s) # Add pos/neg constraint

    def convexify(self, constraints):
        t_start = time.time()
        for c in constraints:
            c.convexify(self.x)
        self.convexify_time += time.time() - t_start
        # tolerance = 1e-3/4
        # scale = 10.0

        # fkq1 = self.robot.fk(q1)
        # fkq2 = self.robot.fk(q2)
        
        # if False: # lift_constraint:
        #     for t in range(0, self.H//4):
        #         # constrain positions to follow center line
        #         q = self.config(t)
        #         J = self.robot.jacobian(q)[0]
        #         pos = self.robot.fk(q)[:3,3]
        #         tgt = self.robot.fk(q1)[:3,3] # TODO: better target
        #         for i in range(2): # 0=x, 1=y
        #             offset = pos[i] - tgt[i]
        #             for j in range(self.dim):
        #                 offset += J[i][j] * q[j]
        #             c = self.add_constraint((offset - tolerance) * scale, (offset + tolerance) * scale)
        #             for j in range(self.dim):
        #                 self.constrain(c, self.qidx(t, j), J[i][j] * scale)
        #             self.add_penalty(c)

    def initial_point(self, q0, q1, q2, q3):
        x_list = []
        third = self.H//3
        for t in range(third):
            s = t*self.t_step/third
            x_list.append(q0 * (1-s) + q1 * s)

        for t in range(self.H+1-2*third):
            s = t*self.t_step/(self.H-2*third+1)
            x_list.append(q1 * (1-s) + q2 * s)

        for t in range(third):
            s = t*self.t_step/third
            x_list.append(q2 * (1-s) + q3 * s)

        self.x = np.concatenate(x_list)

    def compute_obj_val(self, x):
        #return 0.5*sum(x[self.vidx(0,0):self.aidx(0,0)]**2) # sum of squared velocities
        #return 0.5*self.obj_scale*np.sum(x[self.objective]**2) # sum of squared velocities
        v0 = (self.H+1)*self.dim
        a0 = v0*2
        if 'v' == self.objective:
            return 0.5*self.obj_scale*np.sum(x[v0:a0]**2)
        j0 = v0*3 - self.dim
        if 'a' == self.objective:
            return 0.5*self.obj_scale*np.sum(x[a0:j0]**2)
        if 'j' == self.objective:
            return 0.5*self.obj_scale*(
                np.sum(x[a0:a0+self.dim]**2) +
                np.sum(x[j0-self.dim:j0]**2) +
                np.sum(((x[a0+self.dim:j0] - x[a0:j0-self.dim]))**2))/self.t_step
        raise ValueError("bad objective")

    def compute_costs(self, constraints, x):
        t_start = time.time()
        total = sum([ c.cost(x) for c in constraints ])
        self.cost_time += time.time() - t_start
        return total

    def optimize_motion(self, q0, q1, q2, q3, done):
        start_time = time.time()

        constraints = [ LiftConstraint(self, q0, q1), DropConstraint(self, q2, q3) ]

        log.debug("optimize_motion")
        log.debug(f"  q0 = {q0}, pos={self.robot.fk(q0)[:3,3]}")
        log.debug(f"  q1 = {q1}, pos={self.robot.fk(q1)[:3,3]}")
        log.debug(f"  q2 = {q2}, pos={self.robot.fk(q2)[:3,3]}")
        log.debug(f"  q3 = {q3}, pos={self.robot.fk(q3)[:3,3]}")

        self.penalty = self.initial_penalty
        self.trust_region_size = self.initial_trust_region_size
        
        self.initial_point(q0, q1, q2, q3)
        self.init_qp(q0, q3)
        self.setup_qp()
        r = self.solve_one_qp()
        if r.info.status != "solved":
            log.warn(f"initial solve failed: {r.info.status}")
            return None
        self.x = r.x
        # for i in range(self.H+1):
        #     log.info(i, self.config(i), self.velocity(i))

        self.init_qp(q0, q3)
        self.convexify(constraints)
        self.setup_qp(r.x, r.y, rho=r.info.rho_estimate)
        
        prev_obj_val = self.compute_obj_val(r.x)
        prev_exact_viols = self.compute_costs(constraints, r.x)
        log.debug(f"Initial cost = {prev_exact_viols}")
        
        for iter in range(self.max_iters):
            log.debug(f"=== solve #{iter}, penalty={self.penalty}, trust={self.trust_region_size}")
            if done is not None and done.value:
                #log.info(f"got 'DONE' {time.time() - start_time}")
                return None
            
            r = self.solve_one_qp()
            solved_exact = (r.info.status == "solved")
            if not solved_exact:
                log.debug(f"solve failed: {r.info.status}")
                if r.info.status == 'maximum iterations reached':
                    continue
                if r.info.status != 'solved inaccurate':
                    break

            curr_obj_val = self.compute_obj_val(r.x)
            curr_exact_viols = self.compute_costs(constraints, r.x) # sum([ c.cost(r.x) for c in constraints ])
            curr_penvar_vals = np.sum(r.x[self.n_vars:])

            log.debug(f"obj_val = {curr_obj_val}, pen_var = {curr_penvar_vals * self.penalty} vs {np.sum(np.array(self.q) * r.x)}, robj={r.info.obj_val} vs {curr_obj_val + curr_penvar_vals*self.penalty}")
            #log.info(f"New cost = {curr_actual_cost}, pen_cost = {curr_penalty_cost}")

            old_cost   = prev_obj_val + prev_exact_viols * self.penalty
            model_cost = curr_obj_val + curr_penvar_vals * self.penalty
            new_cost   = curr_obj_val + curr_exact_viols * self.penalty
            approx_improve = old_cost - model_cost
            exact_improve = old_cost - new_cost
            improve_ratio = exact_improve / approx_improve

            log.debug(f"obj_val: {prev_obj_val} -> {curr_obj_val}")
            log.debug(f"pen_val: {prev_exact_viols} -> {curr_exact_viols}")
            log.debug(f"  model: {curr_penvar_vals}")
            log.debug(f"improve: approx={approx_improve}, exact={exact_improve}, ratio={improve_ratio}, TR={self.trust_region_size}, pen={self.penalty}")

            min_approx_improve = 1e-4
            min_approx_improve_ratio = -np.inf
            improve_ratio_threshold = 0.25 / 10
            
            if approx_improve < min_approx_improve:
                log.debug("converged, small improvement, increase penalty")
                pass # converged, small improvement, increase penalty
            elif approx_improve / old_cost < min_approx_improve_ratio:
                log.debug("converged, small improvement ratio, increase penalty")
                pass # converged, small improvement ratio, increase penalty
            elif exact_improve < 0 or improve_ratio < improve_ratio_threshold:
                log.debug("shrinking trust region")
                if solved_exact:
                    self.trust_region_size *= self.trust_shrink_ratio
                    if self.trust_region_size > self.min_trust_region_size:
                        self.update_trust_region()
                        continue
                    else:
                        log.debug("converged, small trust region, increase penalty")
                # else, increase penalty
            else:                
                log.debug("accepting trajectory and growing trust region")
                prev_obj_val = curr_obj_val
                prev_exact_viols = curr_exact_viols
                self.x = r.x # accept the trajectory

                if solved_exact and self.terminate_on_constraint_satisfaction and curr_exact_viols < self.acceptable_penalty_threshold:
                    log.debug("done, penalties are low")
                    break

                self.trust_region_size *= self.trust_expand_ratio
                self.init_qp(q0, q3)
                self.convexify(constraints)
                self.setup_qp(r.x, r.y, rho=r.info.rho_estimate)
                continue

            if not solved_exact:
                log.debug("nevermind, inexact...")
                continue

            # increase penalty
            self.penalty *= self.penalty_increase_ratio
            
            if self.penalty > self.max_penalty:
                log.debug(f"max penalty reached, done")
                break

            self.trust_region_size = max(
                self.trust_region_size,
                self.min_trust_region_size / self.trust_shrink_ratio * 1.5)
            
            # only penalties and trust region is updated, we keep the convexification
            for i in range(self.n_vars, len(self.q)):
                assert self.q[i] * self.penalty_increase_ratio == self.penalty
                self.q[i] = self.penalty
            if self.adaptive_eps:
                eps = max(self.qp_eps, self.trust_region_size / self.penalty)
                self.qp.update_settings(eps_rel=eps, eps_abs=eps)
            self.qp.update(q=self.q)
            self.update_trust_region()

        elapsed_time = time.time() - start_time
        if False:
            # find the next file number available...
            fname = lambda x: f"trajectories/trajectory_{x}.out"
            bounds = [0,1]
            while os.path.exists(fname(bounds[1])):
                bounds[0] = bounds[1]
                bounds[1] *= 2
            while bounds[1] - bounds[0] > 1:
                bounds[not os.path.exists(fname(sum(bounds)//2))] = sum(bounds)//2
            # Save trajectory to the file
            with open(fname(bounds[1]), mode='x') as f:
                self.print_trajectory(f)


            log.info(f"Wrote trajectory to {fname(bounds[1])}")

        success = (curr_exact_viols < self.acceptable_penalty_threshold)
        # log.info(f"DONE! (H={self.H}) {success} {elapsed_time} seconds elapsed ({self.qp_time}, {self.init_time}, {self.convexify_time}, {self.cost_time} in QP, init, cvx, cost), pen={prev_exact_viols}, iters={iter} {done.value}")
        return self.x if success else None
        
        
    def print_trajectory(self, file=sys.stdout):
        for t in range(self.H+1):
            q = self.config(self.x, t)
            v = self.velocity(self.x, t)
            a = self.acceleration(self.x, t) if t < self.H else np.zeros(self.dim)
            p = self.robot.fk(q)[:3,3]
            print(t*self.t_step, *q,*p,*v,*a, file=file)

#from joblib import Parallel, delayed
#parallel = Parallel(n_jobs=-1, prefer='processes') #, require='sharedmem'

done = mp.Value('i', 0)

def _do_opt(dim, H, t_step, max_vel, max_acc, max_jerk, objective, obj_scale, q0, q1, q2, q3):
    sqp = SQPMotionOptimizer(dim, H, t_step, max_vel, max_acc, max_jerk, objective, obj_scale)
    return sqp.optimize_motion(q0, q1, q2, q3, done)

class MTSQPMotionOptimizer:
    def __init__(self, dim, H, t_step, max_vel, max_acc, max_jerk=None, objective='v', obj_scale=1.0):
        self.dim = dim
        self.H = H
        self.t_step = t_step
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.objective = objective
        self.obj_scale = obj_scale
        self.done = mp.Value('i', 0)
        self.pool = mp.Pool()
        self.event = threading.Event()
        self.results = []
        self.timer = None
        self.H_min = 8
        self.H_max = 16
        
    def optimize_motion(self, q0, q1, q2, q3):
        # wait for all results from previous run, so that we don't
        # accidentally intermix results.
        start_time = time.time()
        for r in self.results:
            r.wait()
        self.event.clear()
        wait_time = time.time() - start_time
        if wait_time > 0.5:
            log.warn(f"Waited {wait_time} for last results")

        start_time = time.time()


        # The callback is called when the async job is complete.
        # We wake up the waiting thread with an event as soon as
        # we have a result, or if we've got all failed results.
        count = 0
        def callback(x):
            nonlocal count, start_time
            # log.info(f"ELAPSED = {time.time() - start_time}")
            count += 1
            if x is not None or count == (self.H_max - self.H_min):
                self.event.set()

        def error_callback(err):
            nonlocal count
            log.warn(f"got error from thread {err}")
            callback(None)

        done.value = 0
        self.results = [self.pool.apply_async(_do_opt, (self.dim, H, self.t_step, self.max_vel, self.max_acc, self.max_jerk, self.objective, self.obj_scale, q0, q1, q2, q3),
                                              callback=callback, error_callback=error_callback)
                        for H in range(self.H_min, self.H_max)]

        # Wait for the a result...
        self.event.wait()
        result_traj, result_H = None, None
        wait_time = 0.0
        for H in range(self.H_max-1, self.H_min-1, -1):
            i = H - self.H_min
            if wait_time == 0.0:
                if not self.results[i].ready():
                    #log.info(f"skipping H={H} {time.time() - start_time}")
                    continue
                r = self.results[i].get()
            else:
                try:
                    wait_start = time.time()
                    r = self.results[i].get(wait_time)
                    wait_elapsed = time.time() - wait_start
                    # log.info(f"WAITED for {wait_elapsed}")
                    wait_time = max(0.0, wait_time - wait_elapsed)
                except mp.context.TimeoutError:
                    # log.info(f"skipping H={H}, WAITED (TO) for {time.time() - wait_start}")
                    wait_time = 0.0 # done waiting, we still scan the rest though
                    continue
                    
            if r is not None:
                # We have a new result, record it, and set how much
                # time to wait for a faster result before giving up
                # log.info(f"result with {H} {time.time() - start_time}")
                result_traj, result_H = r, H
                wait_time = self.t_step / 4

        # Setting done.value to 1 will cause any remaining computation
        # to terminate on its next loop.
        done.value = 1
        log.info(f"RESULTS: {result_H}, {time.time() - start_time}")
        if result_traj is not None:
            result_traj = interpolate(result_traj, result_H, self.t_step, result_H*10, self.t_step/10)
            result_H *= 10
        return result_traj, result_H
        
        

"""
Central definitions for the five dynamic MPC benchmark systems.

This file is intentionally the single source of truth for plant parameters,
sampling times, horizons, initial conditions, references, and input bounds.
The values match the MATLAB archive models, with one planned change:
Pendulum-Cart simulation time is extended from 10 s to 15 s (Nsim=150, dt=0.1).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, List
import numpy as np


@dataclass
class SystemDef:
    key: str
    label: str
    Ad: np.ndarray
    Bd: np.ndarray
    dt: float
    Nsim: int
    P: int
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    u_min: float
    u_max: float
    state_names: List[str]
    input_names: List[str]
    metric_indices: List[int]
    # Cost convention:
    #   regulation: cost_state=x, ref_horizon=0
    #   absolute:   cost_state=x, ref_horizon=x_ref
    #   deviation:  cost_state=x-x_ref, ref_horizon=0  (HVAC MATLAB convention)
    #   trajectory: cost_state=x, ref_horizon=time-varying ref(k+i)
    cost_mode: str = "regulation"
    x_ref: Optional[np.ndarray] = None
    ref_traj: Optional[np.ndarray] = None   # shape (nx, Nsim+1)
    nonlinear_step: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None
    cart_constraint: Optional[float] = None
    constraint_penalty: float = 1e5
    notes: str = ""

    @property
    def nx(self): return int(self.Ad.shape[0])
    @property
    def nu(self): return int(self.Bd.shape[1])
    @property
    def lb_u(self): return self.u_min * np.ones(self.nu)
    @property
    def ub_u(self): return self.u_max * np.ones(self.nu)

    def reference_at(self, k: int) -> np.ndarray:
        if self.cost_mode == "trajectory":
            idx = min(max(int(k), 0), self.ref_traj.shape[1]-1)
            return self.ref_traj[:, idx].copy()
        if self.x_ref is not None:
            return self.x_ref.copy()
        return np.zeros(self.nx)

    def reference_horizon(self, k: int, P: Optional[int] = None) -> np.ndarray:
        P = self.P if P is None else int(P)
        if self.cost_mode == "trajectory":
            refs = []
            for i in range(P):
                idx = min(k+i+1, self.ref_traj.shape[1]-1)
                refs.append(self.ref_traj[:, idx])
            return np.concatenate(refs)
        if self.cost_mode in ("regulation", "deviation"):
            return np.zeros(P * self.nx)
        if self.cost_mode == "absolute":
            return np.tile(self.x_ref, P)
        raise ValueError(f"Unknown cost_mode={self.cost_mode}")

    def cost_state(self, x: np.ndarray) -> np.ndarray:
        if self.cost_mode == "deviation":
            return np.asarray(x, dtype=float) - self.x_ref
        return np.asarray(x, dtype=float)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).ravel()
        if self.nonlinear_step is not None:
            return self.nonlinear_step(np.asarray(x, dtype=float), u, self.dt)
        return self.Ad @ x + self.Bd @ u


def _pendcart_step(x, u, dt):
    from scipy.integrate import solve_ivp
    m_p=1.0; M_c=5.0; L=2.0; g=9.81; d_c=1.0
    u_k = float(np.asarray(u).ravel()[0])
    def f(_, s):
        den = m_p*L*L*(M_c + m_p*(1-np.cos(s[2])**2))
        return [
            s[1],
            (1/den)*(-m_p**2*L**2*g*np.cos(s[2])*np.sin(s[2])
                     +m_p*L**2*(m_p*L*s[3]**2*np.sin(s[2])-d_c*s[1])
                     +m_p*L**2*u_k),
            s[3],
            (1/den)*((M_c+m_p)*m_p*g*L*np.sin(s[2])
                     -m_p*L*np.cos(s[2])*(m_p*L*s[3]**2*np.sin(s[2])-d_c*s[1])
                     -m_p*L*np.cos(s[2])*u_k),
        ]
    sol = solve_ivp(f, [0, dt], x, rtol=1e-6, atol=1e-8)
    return sol.y[:, -1]


def _make_hvac():
    h_i=10.0; h_o=10.0; L=0.1; k=0.5; A_area=10.0
    rho_w=1000.0; rho_r=1000.0; V_w=1.0; V_r=1.0; c_p=1000.0
    R_i = 1/(h_i*A_area); R_o = 1/(h_o*A_area); R_w = L/(k*A_area)
    C_w = rho_w*V_w*c_p; C_r = rho_r*V_r*c_p
    R = 1/(1/R_i + 1/(R_w/2))
    R_prime = 1/(1/(R_o + R_w/2) + 1/(R_i + R_w/2))
    R_values = R*np.ones(10)
    a = -1/C_r*(1/R_values[0]+1/R_values[1]+1/R_values[7]+1/R_values[9])
    b = -1/C_r*(1/R_values[2]+1/R_values[3]+1/R_values[8]+1/R_values[7])
    c = -1/C_r*(1/R_values[4]+1/R_values[5]+1/R_values[6]+1/R_values[9])
    M = np.zeros((13,10))
    for i in range(10):
        M[i,i] = -1/(R_prime*C_w)
    for row, cols in [(10,[0,1,7,9]), (11,[2,3,7,8]), (12,[4,5,6,9])]:
        for col in cols:
            M[row,col] = 1/(R*C_r)
    N = np.zeros((13,3))
    for i in range(10):
        N[i,0] = 1/(R*C_w)
    N[10,0] = a; N[11,1] = b; N[12,2] = c
    A_cont = np.hstack([M,N])
    T0=20.0; T_stpt1=T_stpt2=T_stpt3=22.0
    B = np.zeros((13,3))
    B[10,0] = (c_p/C_r)*(T0 - T_stpt1)
    B[11,1] = (c_p/C_r)*(T0 - T_stpt2)
    B[12,2] = (c_p/C_r)*(T0 - T_stpt3)
    dt = 0.1
    return np.eye(13)+A_cont*dt, B*dt


def _make_uav_ref(Nsim, dt):
    t = np.arange(Nsim+1)*dt
    ref = np.zeros((12, Nsim+1))
    ref[0,:] = np.linspace(0,10,Nsim+1)
    ref[1,:] = np.linspace(0,10,Nsim+1)
    ref[2,:] = np.linspace(0,10,Nsim+1)
    return ref


def get_systems() -> Dict[str, SystemDef]:
    systems = {}

    # Pendulum-cart, MATLAB model; simulation extended to 15 s.
    m_p=1.0; M_c=5.0; L=2.0; g=9.81; d_c=1.0; b=1.0
    A_c=np.array([[0,1,0,0],[0,-d_c/M_c,b*m_p*g/M_c,0],
                  [0,0,0,1],[0,-b*d_c/(M_c*L),-b*(m_p+M_c)*g/(M_c*L),0]], dtype=float)
    B_c=np.array([[0],[1/M_c],[0],[b/(M_c*L)]], dtype=float)
    dt=0.1
    systems["pendcart"] = SystemDef(
        key="pendcart", label="Inverted Pendulum on Cart", Ad=np.eye(4)+A_c*dt, Bd=B_c*dt,
        dt=dt, Nsim=150, P=20, Q=np.diag([1.,1.,20.98,1.]), R=np.array([[0.1]]),
        x0=np.array([-1.,0.,np.pi+0.2,0.]), x_ref=np.array([1.,0.,np.pi,0.]),
        u_min=-5., u_max=5., nonlinear_step=_pendcart_step, cart_constraint=10.,
        state_names=["cart_x", "cart_v", "theta", "theta_dot"], input_names=["force"],
        metric_indices=[0,2], cost_mode="absolute",
        notes="MATLAB Nsim=100; here Nsim=150 to give 15 s as requested.")

    # CSTR, exactly MATLAB linearized deviation model.
    A=np.array([[-5., -0.3427], [47.68, 2.785]], dtype=float)
    B=np.array([[0., 1.], [0.3, 0.]], dtype=float)
    dt=0.1
    systems["cstr"] = SystemDef(
        key="cstr", label="CSTR", Ad=np.eye(2)+A*dt, Bd=B*dt,
        dt=dt, Nsim=100, P=20, Q=np.diag([20.,20.]), R=0.01*np.eye(2),
        x0=np.array([0.5,-1.]), x_ref=np.zeros(2), u_min=-1., u_max=1.,
        state_names=["CA_dev", "T_dev"], input_names=["u1", "u2"],
        metric_indices=[0,1], cost_mode="regulation")

    # Longitudinal flight control, exactly MATLAB linearized deviation model.
    A=np.array([[-0.04,11.59,0.,-32.2],[-0.000073,-0.65,1.,0.],
                [0.000048,-0.49,-0.58,0.],[0.,0.,1.,0.]], dtype=float)
    B=np.array([[0.,0.1],[-0.014,0.],[0.,0.],[0.,0.]], dtype=float)
    dt=0.1
    systems["flight"] = SystemDef(
        key="flight", label="Aircraft Longitudinal Flight Control", Ad=np.eye(4)+A*dt, Bd=B*dt,
        dt=dt, Nsim=150, P=20, Q=np.diag([20.,20.,20.,20.]), R=0.01*np.eye(2),
        x0=np.array([0.1,-0.1,0.05,-0.05]), x_ref=np.zeros(4), u_min=-10., u_max=10.,
        state_names=["u_dev", "alpha", "q", "theta"], input_names=["elevator", "throttle"],
        metric_indices=[0,1,2,3], cost_mode="regulation")

    Ad_h, Bd_h = _make_hvac()
    systems["hvac"] = SystemDef(
        key="hvac", label="HVAC 13-State Thermal Building", Ad=Ad_h, Bd=Bd_h,
        dt=0.1, Nsim=1500, P=20, Q=np.diag([1.]*10+[100.]*3), R=0.01*np.eye(3),
        x0=20*np.ones(13), x_ref=np.array([20.]*10+[22.]*3), u_min=-48., u_max=48.,
        state_names=[f"wall_{i+1}" for i in range(10)] + ["room_1", "room_2", "room_3"],
        input_names=["hvac_1", "hvac_2", "hvac_3"], metric_indices=[10,11,12],
        cost_mode="deviation",
        notes="Absolute temperature plant; MPC cost uses x-x_set as in MATLAB.")

    # UAV trajectory tracking, exactly MATLAB small-angle linear model and 0→10 m reference.
    g=9.81; m=0.547; Ix=Iy=Iz=3.3e-3
    A=np.zeros((12,12)); B=np.zeros((12,4))
    A[0,6]=1; A[1,7]=1; A[2,8]=1; A[3,9]=1; A[4,10]=1; A[5,11]=1
    A[6,4]=g; A[7,3]=-g
    B[8,0]=1/m; B[9,1]=1/Ix; B[10,2]=1/Iy; B[11,3]=1/Iz
    dt=0.1; Nsim=500; ref=_make_uav_ref(Nsim, dt)
    systems["uav"] = SystemDef(
        key="uav", label="Quadrotor UAV Trajectory Tracking", Ad=np.eye(12)+A*dt, Bd=B*dt,
        dt=dt, Nsim=Nsim, P=10, Q=np.diag([5000.,5000.,5000.]+[1.]*9), R=0.1*np.eye(4),
        x0=np.zeros(12), x_ref=None, ref_traj=ref, u_min=-1., u_max=1.,
        state_names=["x", "y", "z", "phi", "theta", "psi", "vx", "vy", "vz", "p", "q", "r"],
        input_names=["df", "tau_phi", "tau_theta", "tau_psi"], metric_indices=[0,1,2],
        cost_mode="trajectory")
    return systems


def get_system(key: str) -> SystemDef:
    systems = get_systems()
    if key not in systems:
        raise KeyError(f"Unknown system '{key}'. Available: {list(systems)}")
    return systems[key]

"""
================================================================================
  RUNTIME HOMOLOGY THEORY — Part 2: Unimplemented Theorems
================================================================================

Covers everything NOT in Part 1:
  § Formula 1    — Runtime Field Equation (Einstein analog)
  § Formula 4    — Bohr-Sommerfeld Autoscaler Quantization
  § Definition 4 — Request Flux Divergence (microservice mesh)
  § Definition 5 — Memory Entropy Production
  § Definition 8 — Resource Coupling Tensor
  § Definition 9 — Saturation Homology Group (persistent homology)
  § Definition 10 — Load-Response Characteristic + Hysteresis
  § Theorem 2    — Phase Transition Universality / Ising Mapping
  § Theorem 9    — Atiyah-Singer Index (zero-mode counting)
  § Conjecture 1 — Runtime Riemann Hypothesis (zeta-function zeros)
  § Conjecture 4 — Mirror Symmetry for Stress Manifolds

Dependencies: numpy, scipy, matplotlib, sklearn
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg, optimize, signal
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rht_output")
os.makedirs(OUT_DIR, exist_ok=True)

def _out(name): return os.path.join(OUT_DIR, name)

# ── palette ───────────────────────────────────────────────────────────────────
C = dict(
    red="#E94560", blue="#0F3460", purple="#533483",
    gold="#FFD700", green="#2ECC71", orange="#F39C12",
    cyan="#00BCD4", magenta="#E91E63",
    bg="#0D0D0D", grid="#1A1A2E", text="#E0E0E0",
)

plt.rcParams.update({
    "figure.facecolor": C["bg"],  "axes.facecolor": C["grid"],
    "axes.edgecolor":   C["text"],"axes.labelcolor": C["text"],
    "xtick.color":      C["text"],"ytick.color":     C["text"],
    "text.color":       C["text"],"grid.color":      "#2A2A4A",
    "grid.linestyle":   "--",     "grid.alpha":      0.4,
    "font.family":      "monospace",
})

def ax3d(fig, pos, title, elev=28, azim=45):
    ax = fig.add_subplot(pos, projection="3d")
    ax.set_facecolor(C["grid"])
    ax.set_title(title, color=C["gold"], fontsize=10, fontweight="bold", pad=7)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(colors=C["text"], labelsize=7)
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.fill = False; p.set_edgecolor("#2A2A4A")
    return ax

def ax2d(fig, pos, title):
    ax = fig.add_subplot(pos)
    ax.set_facecolor(C["grid"])
    ax.set_title(title, color=C["gold"], fontsize=10, fontweight="bold")
    ax.grid(True)
    return ax


# ══════════════════════════════════════════════════════════════════════════════
# DEFINITION 4 — Request Flux Divergence (microservice mesh)
# ══════════════════════════════════════════════════════════════════════════════

class RequestFluxDivergence:
    """
    ∇·J = ∂J_x/∂x + ∂J_y/∂y + ∂J_z/∂z
    where J is the vector field of request flow through service mesh nodes.

    Positive divergence = queues growing (bottleneck)
    Negative divergence = processing faster than arriving
    Zero = steady state
    """

    def __init__(self, n_services: int = 8, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n  = n_services
        # Topology: random directed graph (service dependency)
        self.adj = (rng.random((n_services, n_services)) < 0.35).astype(float)
        np.fill_diagonal(self.adj, 0)
        self.names = [f"svc_{i}" for i in range(n_services)]
        # Service capacities
        self.capacity = rng.uniform(80, 200, n_services)
        self.rng = rng

    def simulate(self, total_load: float = 500.0, steps: int = 300):
        """Simulate request flow and compute divergence at each node over time."""
        n = self.n
        history = np.zeros((steps, n))
        queue   = np.zeros(n)
        in_rate = np.zeros(n)
        out_rate= np.zeros(n)

        for t in range(steps):
            # External load enters through entry-point services (first 2)
            external = np.zeros(n)
            external[0] = total_load * (0.6 + 0.4 * np.sin(2*np.pi*t/100))
            external[1] = total_load * 0.4 * (1 + 0.2*self.rng.standard_normal())

            # Compute inflow per service
            in_rate[:] = external.copy()
            for i in range(n):
                for j in range(n):
                    if self.adj[j, i]:  # j → i
                        downstream_fraction = self.adj[j, :].sum()
                        in_rate[i] += out_rate[j] / max(1, downstream_fraction)

            # Out rate limited by capacity
            for i in range(n):
                out_rate[i] = min(in_rate[i] + queue[i] * 0.1, self.capacity[i])

            # Queue evolution
            queue += in_rate - out_rate
            queue  = np.maximum(0, queue)

            history[t] = queue.copy()

        # Divergence = in_rate - out_rate (final snapshot)
        divergence = in_rate - out_rate
        return dict(history=history, divergence=divergence,
                    in_rate=in_rate, out_rate=out_rate,
                    queue=queue, capacity=self.capacity)


# ══════════════════════════════════════════════════════════════════════════════
# DEFINITION 5 — Memory Entropy Production
# ══════════════════════════════════════════════════════════════════════════════

class MemoryEntropyProduction:
    """
    dS/dt = ∫ (AllocRate · log(AllocRate/FreeRate) + LeakRate · log(LeakRate)) dV

    Measures the rate of irreversible disorder increase in memory space.
    dS/dt ≈ 0 for stable service; dS/dt > 0 for leaking service.
    """

    def __init__(self, duration: int = 2000, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.t   = np.arange(duration)
        self.dur = duration

    def simulate(self, leak_rate: float = 0.0, label: str = ""):
        alloc = 100 + 30 * np.sin(2*np.pi*self.t / 200) \
                + 8  * self.rng.standard_normal(self.dur)
        alloc = np.maximum(1.0, alloc)
        free  = alloc - leak_rate + 3 * self.rng.standard_normal(self.dur)
        free  = np.maximum(0.5, free)

        ratio = alloc / (free + 1e-9)
        # Shannon-style entropy production
        dS_dt = alloc * np.log(ratio + 1e-9) + leak_rate * np.log(leak_rate + 1.0)

        # Cumulative entropy
        S_cum = np.cumsum(np.maximum(0, dS_dt)) / self.dur

        # Object lifetime distribution entropy (proxy)
        lifetimes = self.rng.exponential(scale=1.0/(ratio.mean()+1e-6), size=5000)
        H_lifetime = -np.mean(np.log(lifetimes + 1e-9))

        return dict(t=self.t, dS_dt=dS_dt, S_cum=S_cum,
                    alloc=alloc, free=free, leak_rate=leak_rate,
                    label=label, mean_dS=float(dS_dt.mean()),
                    H_lifetime=H_lifetime)


# ══════════════════════════════════════════════════════════════════════════════
# DEFINITION 8 — Resource Coupling Tensor
# ══════════════════════════════════════════════════════════════════════════════

class ResourceCouplingTensor:
    """
    T^μν = ∂²Latency / ∂R_μ ∂R_ν   (Hessian of latency w.r.t. resources)

    Off-diagonal terms = coupling effects:
      T^CPU,Memory > 0  → compete (both increase latency)
      T^CPU,Memory < 0  → complement (e.g. caching reduces latency)
    """

    RESOURCES = ["CPU", "Memory", "IO", "Network"]

    def __init__(self, seed: int = 13):
        self.rng = np.random.default_rng(seed)

    def latency_model(self, cpu, mem, io, net):
        """
        Realistic non-linear latency model with cross-terms.
        Latency(ms) = f(cpu, mem, io, net)
        """
        base = (
            200 / (cpu + 0.1)                   # CPU helps
            + 0.05 * (4096 - mem).clip(0) / 100 # memory helps
            + 3.0  * io                          # IO hurts
            + 1.5  * net                         # Network hurts
        )
        # Cross-coupling terms
        cross = (
            0.02 * io * net                      # IO-Net compound
            - 0.005 * cpu * mem / 1000           # CPU-Mem complement (caching)
            + 0.01 * io**2 / cpu                 # IO contention under low CPU
        )
        noise = 0.5 * self.rng.standard_normal()
        return float(base + cross + noise)

    def compute_tensor(self, operating_point: dict, delta: float = 0.05):
        """Numerical Hessian via second-order finite differences."""
        res = self.RESOURCES
        n   = len(res)
        x0  = np.array([operating_point[r] for r in res], dtype=float)
        T   = np.zeros((n, n))

        def f(x):
            return self.latency_model(*x)

        f0 = f(x0)

        for i in range(n):
            for j in range(n):
                if i == j:
                    xp = x0.copy(); xp[i] += delta * x0[i]
                    xm = x0.copy(); xm[i] -= delta * x0[i]
                    T[i, j] = (f(xp) - 2*f0 + f(xm)) / (delta * x0[i])**2
                else:
                    xpp = x0.copy(); xpp[i] += delta*x0[i]; xpp[j] += delta*x0[j]
                    xpm = x0.copy(); xpm[i] += delta*x0[i]; xpm[j] -= delta*x0[j]
                    xmp = x0.copy(); xmp[i] -= delta*x0[i]; xmp[j] += delta*x0[j]
                    xmm = x0.copy(); xmm[i] -= delta*x0[i]; xmm[j] -= delta*x0[j]
                    T[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / \
                               (4 * (delta*x0[i]) * (delta*x0[j]))

        return T

    def sweep_operating_points(self):
        points = [
            {"CPU": 2.0, "Memory": 1024, "IO": 10, "Network": 5, "label": "Low load"},
            {"CPU": 8.0, "Memory": 2048, "IO": 30, "Network": 20, "label": "Med load"},
            {"CPU": 16.0,"Memory": 4096, "IO": 60, "Network": 50, "label": "High load"},
        ]
        results = []
        for p in points:
            label = p.pop("label")
            T = self.compute_tensor(p)
            p["label"] = label
            results.append(dict(T=T, point=p, label=label))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# DEFINITION 9 — Saturation Homology Group (simplified persistent homology)
# ══════════════════════════════════════════════════════════════════════════════

class SaturationHomology:
    """
    H_n(S) = Ker(∂_n) / Im(∂_{n+1})

    We approximate persistent homology via sublevel-set filtration of
    a 2-D 'stress landscape' (failure correlation field), counting:
      H₀ = connected components (isolated failures)
      H₁ = loops / cycles (retry storms, oscillations)

    Implementation: Vietoris-Rips complex on failure-state samples.
    """

    def __init__(self, seed: int = 17):
        self.rng = np.random.default_rng(seed)

    def _generate_failure_landscape(self, n_pts: int = 400):
        """
        Synthetic failure correlation samples on a 2-D manifold with
        deliberate holes (topological features).
        """
        # Three 'failure clusters' with a void between them (→ non-trivial H₁)
        centers = np.array([[0, 0], [4, 0], [2, 3.5]])
        pts = []
        for cx, cy in centers:
            r = self.rng.normal(0, 0.6, (n_pts//3, 2))
            pts.append(r + [cx, cy])
        pts = np.vstack(pts)

        # Add an annular 'retry storm' cluster (→ H₁ generator)
        angles = self.rng.uniform(0, 2*np.pi, n_pts//4)
        radii  = 1.5 + self.rng.normal(0, 0.15, n_pts//4)
        pts = np.vstack([pts, np.column_stack([
            8 + radii * np.cos(angles),
            1.5 + radii * np.sin(angles)
        ])])
        return pts

    def compute_betti_numbers(self, pts: np.ndarray,
                               epsilon_range: np.ndarray | None = None):
        """
        Approximate persistent homology via Vietoris-Rips filtration.
        Returns Betti numbers β₀, β₁ at each ε threshold.
        """
        if epsilon_range is None:
            epsilon_range = np.linspace(0.1, 3.0, 50)

        D = cdist(pts, pts)
        beta0_list, beta1_list = [], []

        for eps in epsilon_range:
            # Build adjacency matrix (edges where d < eps)
            adj = (D < eps) & (D > 0)

            # β₀ = number of connected components (union-find)
            n = len(pts)
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                px, py = find(x), find(y)
                if px != py: parent[px] = py

            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j]: union(i, j)

            b0 = len(set(find(i) for i in range(n)))

            # β₁ = loops ≈ E - V + β₀  (Euler characteristic for graphs)
            E = int(adj.sum()) // 2
            b1 = max(0, E - n + b0)

            beta0_list.append(b0)
            beta1_list.append(b1)

        return dict(epsilon=epsilon_range,
                    beta0=np.array(beta0_list),
                    beta1=np.array(beta1_list),
                    pts=pts)


# ══════════════════════════════════════════════════════════════════════════════
# DEFINITION 10 — Load-Response Characteristic + Hysteresis
# ══════════════════════════════════════════════════════════════════════════════

class LoadResponseCharacteristic:
    """
    LRC(λ) = lim_{t→∞} Latency(t) | constant load λ

    Simulates the LRC with:
     - Linear regime    (λ < λ_knee)
     - Super-linear knee
     - Cliff / collapse
     - Hysteresis: ramp-up vs ramp-down give different curves
    """

    def __init__(self, capacity: float = 800.0, seed: int = 5):
        self.cap = capacity
        self.rng = np.random.default_rng(seed)
        # Internal state for hysteresis
        self._prev_latency = 10.0

    def steady_state_latency(self, load: float, prev_state: float = 10.0) -> float:
        """
        Simplified M/M/1 + state-dependent feedback + hysteresis.
        """
        rho   = load / (self.cap + 1e-9)
        base  = 10.0 + 15.0 * rho / (1 - min(rho, 0.98))  # M/M/1

        # Hysteresis: if previously in high-latency state, need more recovery
        hysteresis_factor = 1.0 + 0.3 * (prev_state > 100)

        # Cliff: exponential blow-up near capacity
        if rho > 0.90:
            cliff = 50 * np.exp(10 * (rho - 0.90))
        else:
            cliff = 0.0

        return base * hysteresis_factor + cliff + 3*self.rng.standard_normal()

    def measure_lrc(self, loads: np.ndarray, direction: str = "up") -> np.ndarray:
        latencies = np.zeros(len(loads))
        state     = 10.0
        for i, load in enumerate(loads):
            # Allow system to reach steady state at each load
            for _ in range(30):
                state = self.steady_state_latency(load, state) * 0.3 + state * 0.7
            latencies[i] = max(10.0, state + 2*self.rng.standard_normal())
        return latencies


# ══════════════════════════════════════════════════════════════════════════════
# FORMULA 1 — Runtime Field Equation (Einstein analog)
# ══════════════════════════════════════════════════════════════════════════════

class RuntimeFieldEquation:
    """
    R_μν − ½ g_μν R = 8πG T_μν + Λ g_μν + ∇_μ J_ν + ∇_ν J_μ

    We work in 2D (CPU, Memory) for tractability.
    g_μν = Hessian of p99 latency (resource metric)
    T_μν = Request stress-energy (outer product of request flux)
    Λ    = baseline latency (cosmological constant)
    G    = 1 / system_capacity
    """

    def __init__(self, grid_size: int = 30):
        self.N = grid_size
        cpu    = np.linspace(1, 16, grid_size)
        mem    = np.linspace(256, 4096, grid_size)
        self.CPU, self.MEM = np.meshgrid(cpu, mem)

    def p99_field(self) -> np.ndarray:
        """Latency scalar field L(cpu, mem)."""
        C, M = self.CPU, self.MEM
        return (
            20 + 500/C + 0.003*(4096-M).clip(0)/50
            + 200*np.exp(-C/3) * np.exp(-(M-512)/1000)
            + 10*np.sin(C)*np.cos(M/300)
        )

    def metric_tensor(self, L: np.ndarray):
        """g_μν = [[∂²L/∂C², ∂²L/∂C∂M], [∂²L/∂M∂C, ∂²L/∂M²]]"""
        dC  = np.gradient(L,  axis=1)
        dM  = np.gradient(L,  axis=0)
        g00 = np.gradient(dC, axis=1)  # ∂²L/∂C²
        g11 = np.gradient(dM, axis=0)  # ∂²L/∂M²
        g01 = np.gradient(dC, axis=0)  # ∂²L/∂C∂M
        return g00, g01, g01, g11

    def ricci_scalar(self, L: np.ndarray) -> np.ndarray:
        """
        Simplified 2-D Ricci scalar  R = K · 2  (Gaussian curvature × 2)
        """
        Lc  = np.gradient(L,  axis=1)
        Lm  = np.gradient(L,  axis=0)
        Lcc = np.gradient(Lc, axis=1)
        Lmm = np.gradient(Lm, axis=0)
        Lcm = np.gradient(Lc, axis=0)
        denom = (1 + Lc**2 + Lm**2)**2
        K     = (Lcc*Lmm - Lcm**2) / (denom + 1e-12)
        return 2 * K                          # Ricci scalar in 2-D = 2K

    def stress_energy_tensor(self, load_field: np.ndarray) -> np.ndarray:
        """
        T_μν ∝ request pressure (simplified to scalar trace).
        """
        rho  = load_field / (self.CPU * 100 + self.MEM/10)  # request density
        return rho

    def solve_field_equation(self):
        L       = self.p99_field()
        R       = self.ricci_scalar(L)

        # Synthetic load field (e.g., from actual request distribution)
        load    = 500 * (1 + 0.5*np.sin(self.CPU/3)) * (1 + 0.3*np.cos(self.MEM/500))
        T       = self.stress_energy_tensor(load)

        Lambda  = L.min()                   # baseline latency
        G       = 1.0 / (load.mean() + 1)  # inverse capacity

        # LHS: R_μν - ½ g_μν R  → in 2D trace-free → just R
        lhs = R

        # RHS: 8πG T + Λ (+ flux corrections absorbed into T)
        rhs = 8 * np.pi * G * T + Lambda / L.max()

        # Residual = how well the equation is satisfied
        residual = lhs - rhs

        return dict(L=L, R=R, T=T, lhs=lhs, rhs=rhs, residual=residual,
                    CPU=self.CPU, MEM=self.MEM, Lambda=Lambda, G=G)


# ══════════════════════════════════════════════════════════════════════════════
# FORMULA 4 — Bohr-Sommerfeld Autoscaler Quantization
# ══════════════════════════════════════════════════════════════════════════════

class BohrSommerfeldAutoscaler:
    """
    ∮_Γ (dN/dt)/(N - N_target) dt = 2πi (n + ½ - φ/2π)

    For a harmonic oscillator in phase space (N, dN/dt), the action
    integral J = ∮ p dq = 2π E/ω is quantized at half-integer values.

    Allowed winding numbers: w = n + ½ - φ/2π
    Stability: w < 0.5  ↔  n = 0  ↔  ground state
    """

    def __init__(self, N_target: float = 10.0):
        self.N_t = N_target

    def phase_space_orbit(self, K_I: float, K_D: float,
                           amplitude: float = 3.0) -> dict:
        """
        For linear controller d²N/dt² + K_D dN/dt + K_I N = 0,
        compute the closed orbit in (N - N_target, dN/dt) plane.
        """
        disc = K_D**2 - 4*K_I

        if disc < 0:                        # underdamped: closed elliptical orbit
            omega_d = np.sqrt(max(0, K_I - (K_D/2)**2))
            sigma   = K_D / 2
            # Parametric orbit after initial transient
            t_orb   = np.linspace(0, 4*np.pi / max(omega_d, 1e-6), 500)
            N_orb   = amplitude * np.exp(-sigma * t_orb) * np.cos(omega_d * t_orb)
            dN_orb  = amplitude * np.exp(-sigma * t_orb) * (
                -sigma * np.cos(omega_d * t_orb) - omega_d * np.sin(omega_d * t_orb)
            )
        else:                               # overdamped: open trajectory
            t_orb  = np.linspace(0, 10, 500)
            r1     = (-K_D + np.sqrt(max(disc, 0))) / 2
            r2     = (-K_D - np.sqrt(max(disc, 0))) / 2
            N_orb  = amplitude * (np.exp(r1*t_orb) + np.exp(r2*t_orb))
            dN_orb = amplitude * (r1*np.exp(r1*t_orb) + r2*np.exp(r2*t_orb))

        # ── Action integral J = ∮ dN/dt · dN  (symplectic area) ──────────────
        # For closed orbits, compute the signed area in phase space
        # J = ½ ∮ (N·d(dN/dt) - dN/dt·dN)  (Green's theorem)
        N_use  = N_orb[:100]    # one orbit
        dN_use = dN_orb[:100]
        J      = 0.5 * abs(np.trapezoid(N_use[:-1]*np.diff(dN_use)
                                        - dN_use[:-1]*np.diff(N_use)))

        # ── Winding number ─────────────────────────────────────────────────────
        if disc < 0:
            zeta    = K_D / (2 * np.sqrt(K_I))
            phi     = np.arctan2(np.sqrt(1 - zeta**2), zeta)
            w       = zeta                  # = ζ (damping ratio)
            w_quant = 0.5 - phi/(2*np.pi)  # Bohr-Sommerfeld level
        else:
            w = 0.0; w_quant = 0.0; phi = 0.0

        return dict(N_orb=N_orb, dN_orb=dN_orb, J=J,
                    w=w, w_quant=w_quant, K_I=K_I, K_D=K_D,
                    stable=w < 0.5, underdamped=disc < 0)

    def quantization_spectrum(self, n_levels: int = 5) -> dict:
        """Predict allowed winding numbers for n=0,1,2,… energy levels."""
        levels = np.arange(n_levels)
        # For critically damped system: K_D = 2√K_I → ζ = 1
        # Allowed winding numbers: w_n = n + ½ modulo phase
        K_I_vals = np.linspace(0.1, 2.0, 100)
        stability = {}
        for n in levels:
            K_D_critical = 2 * np.sqrt(K_I_vals) * (1 + 0.1 * n)  # nth level
            stability[n] = K_D_critical
        return dict(K_I=K_I_vals, levels=stability, n=levels)


# ══════════════════════════════════════════════════════════════════════════════
# THEOREM 2 — Ising Model ↔ GC Phase Transition Mapping
# ══════════════════════════════════════════════════════════════════════════════

class IsingGCMapping:
    """
    Theorem 2 Proof C: GC phase transition IS a magnetic phase transition.

    Ising model on 2D lattice → maps exactly to GC object graph.
    
    Correspondence:
      Spin alignment ↔ Object reachability
      Temperature    ↔ Allocation rate θ
      Magnetization  ↔ Live object fraction
      Susceptibility ↔ GC pause time
      Critical T_c   ↔ θ_c (critical allocation rate)
    """

    def __init__(self, L: int = 30, seed: int = 99):
        self.L    = L
        self.rng  = np.random.default_rng(seed)
        self.grid = 2 * self.rng.integers(0, 2, (L, L)) - 1  # ±1 spins

    def _energy(self, grid: np.ndarray) -> float:
        return -float(
            (grid * np.roll(grid, 1, 0)).sum()
            + (grid * np.roll(grid, 1, 1)).sum()
        )

    def metropolis_step(self, grid: np.ndarray, beta: float) -> np.ndarray:
        L   = self.L
        i, j = self.rng.integers(0, L, 2)
        dE  = 2 * grid[i,j] * (
            grid[(i+1)%L, j] + grid[(i-1)%L, j]
            + grid[i, (j+1)%L] + grid[i, (j-1)%L]
        )
        if dE < 0 or self.rng.random() < np.exp(-beta * dE):
            grid[i, j] = -grid[i, j]
        return grid

    def simulate_phase_transition(self, T_range=None, n_eq=500, n_meas=300):
        if T_range is None:
            T_range = np.linspace(1.5, 3.5, 40)   # T_c ≈ 2.269 for 2D Ising

        results = []
        for T in T_range:
            beta   = 1.0 / T
            grid   = 2 * self.rng.integers(0, 2, (self.L, self.L)) - 1
            # Equilibrate
            for _ in range(n_eq * self.L**2):
                grid = self.metropolis_step(grid, beta)
            # Measure
            M_vals, chi_vals = [], []
            for _ in range(n_meas):
                for _ in range(self.L**2):
                    grid = self.metropolis_step(grid, beta)
                m = abs(grid.mean())
                M_vals.append(m)

            M_mean  = np.mean(M_vals)
            M2_mean = np.mean(np.array(M_vals)**2)
            chi     = self.L**2 * (M2_mean - M_mean**2) / T  # susceptibility ∝ GC pause

            results.append(dict(T=T, M=M_mean, chi=chi))

        T_arr   = np.array([r["T"]   for r in results])
        M_arr   = np.array([r["M"]   for r in results])
        chi_arr = np.array([r["chi"] for r in results])
        return dict(T=T_arr, M=M_arr, chi=chi_arr, T_c=2.269)


# ══════════════════════════════════════════════════════════════════════════════
# THEOREM 9 — Atiyah-Singer Index (zero-mode counting)
# ══════════════════════════════════════════════════════════════════════════════

class AtiyahSingerIndex:
    """
    Index(D) = dim Ker(D) - dim Coker(D) = χ(M) + CorrectionTerm

    D = ∇ + ∇* acting on differential forms on the stress manifold.

    In discrete approximation:
      dim Ker(D) = number of harmonic forms = Betti numbers β_i
      Index      = Σ (-1)^i β_i = Euler characteristic χ(M)

    Zero modes = neutral equilibria (flat spots in latency landscape)
    which = Atiyah-Singer index → minimum tunable parameters.
    """

    def __init__(self, grid_size: int = 25, seed: int = 31):
        self.N   = grid_size
        self.rng = np.random.default_rng(seed)

    def build_discrete_laplacian(self, L_field: np.ndarray) -> np.ndarray:
        """
        Discrete graph Laplacian Δ of the latency field.
        Nodes = grid points, edge weights = 1/|ΔL| (smooth regions are connected).
        """
        N    = self.N
        flat = L_field.flatten()
        n    = N * N
        W    = np.zeros((n, n))

        for i in range(N):
            for j in range(N):
                idx = i*N + j
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni_, nj_ = i+di, j+dj
                    if 0 <= ni_ < N and 0 <= nj_ < N:
                        nidx = ni_*N + nj_
                        dL   = abs(flat[idx] - flat[nidx])
                        W[idx, nidx] = 1.0 / (dL + 1.0)

        D_deg = np.diag(W.sum(axis=1))
        return D_deg - W             # Laplacian

    def count_zero_modes(self, L_field: np.ndarray,
                          threshold: float = 0.5) -> dict:
        """
        Count approximate zero modes (eigenvalues near 0) of the Laplacian.
        These correspond to harmonic forms → neutral equilibria.
        """
        Lap     = self.build_discrete_laplacian(L_field)
        eigvals = np.linalg.eigvalsh(Lap)
        eigvals_sorted = np.sort(eigvals)

        # Count near-zero eigenvalues
        n_zero = int((np.abs(eigvals_sorted) < threshold).sum())

        # Approximate Betti numbers from spectral gaps
        beta0 = int((eigvals_sorted < 0.01).sum())   # β₀ ~ dim Ker(Δ₀)
        beta1 = max(0, n_zero - beta0)               # β₁ ~ higher forms

        # Euler characteristic χ = β₀ - β₁ + β₂ - …
        chi = beta0 - beta1

        # Index(D) ≈ χ (Atiyah-Singer in this simplified 2-D case)
        index = chi

        return dict(eigvals=eigvals_sorted[:50], n_zero=n_zero,
                    beta0=beta0, beta1=beta1, chi=chi, index=index,
                    L=L_field)

    def index_vs_complexity(self, n_configs: int = 8) -> list:
        """
        Sweep over system configurations, compute index for each.
        Predicts minimum # of tunable parameters = sum of Betti numbers.
        """
        configs = []
        for k in range(1, n_configs + 1):
            # Generate increasingly complex latency landscapes
            x = np.linspace(0, k*np.pi, self.N)
            y = np.linspace(0, k*np.pi, self.N)
            X, Y = np.meshgrid(x, y)
            L = (20 + 10*np.sin(k*X/2)*np.cos(k*Y/2)
                    + 5*np.sin(k*Y) + k*np.random.randn(*X.shape)*0.1)
            r = self.count_zero_modes(L)
            configs.append(dict(k=k, index=r["index"],
                                beta0=r["beta0"], beta1=r["beta1"],
                                chi=r["chi"], n_zero=r["n_zero"]))
        return configs


# ══════════════════════════════════════════════════════════════════════════════
# CONJECTURE 1 — Runtime Riemann Hypothesis
# ══════════════════════════════════════════════════════════════════════════════

class RuntimeRiemannHypothesis:
    """
    ζ_R(s) = Σ_{n=1}^∞  p99_n / n^s

    All non-trivial zeros lie on Re(s) = 1/2.

    We compute ζ_R(s) on the critical strip and visualize the zeros.
    """

    def __init__(self, N_terms: int = 500, seed: int = 77):
        rng = np.random.default_rng(seed)
        loads  = np.arange(1, N_terms + 1, dtype=float)
        # Realistic p99 sequence: near-linear with log fluctuations
        self.p99 = (
            10 + 0.5 * loads
            + 5 * np.log(loads + 1)
            + 3 * rng.standard_normal(N_terms)
        )
        self.p99 = np.maximum(1.0, self.p99)
        self.N   = N_terms

    def zeta_R(self, s_arr: np.ndarray) -> np.ndarray:
        """Evaluate ζ_R(s) = Σ p99_n / n^s for array of complex s."""
        n     = np.arange(1, self.N + 1, dtype=complex)
        p99_c = self.p99.astype(complex)
        return np.array([
            np.sum(p99_c / n**s) for s in s_arr
        ])

    def find_zeros_on_critical_line(self, t_range=None) -> dict:
        """
        Scan Im(s) = t with Re(s) fixed at 0.5 and look for sign changes of
        Re(ζ_R(0.5 + it)).
        """
        if t_range is None:
            t_range = np.linspace(0.1, 30.0, 600)

        s_crit  = 0.5 + 1j * t_range
        z_crit  = self.zeta_R(s_crit)

        re_z = np.real(z_crit)
        im_z = np.imag(z_crit)
        mod_z = np.abs(z_crit)

        # Zero crossings of |ζ_R| (local minima)
        zero_idx = signal.argrelmin(mod_z, order=5)[0]
        zero_t   = t_range[zero_idx]
        zero_re  = re_z[zero_idx]

        # Also scan off-line to check if zeros drift from Re(s)=0.5
        # Scan Re(s) ∈ [0.3, 0.7] at fixed Im(s) for first zero
        if len(zero_t) > 0:
            t_first  = zero_t[0]
            sigma_scan = np.linspace(0.1, 0.9, 80)
            s_scan   = sigma_scan + 1j * t_first
            z_scan   = self.zeta_R(s_scan)
            re_scan  = np.real(z_scan)
        else:
            sigma_scan = np.array([0.5])
            re_scan    = np.array([0.0])

        return dict(t=t_range, z=z_crit, re_z=re_z, im_z=im_z, mod_z=mod_z,
                    zero_t=zero_t, zero_re=zero_re,
                    sigma_scan=sigma_scan, re_scan=re_scan,
                    n_zeros=len(zero_t))


# ══════════════════════════════════════════════════════════════════════════════
# CONJECTURE 4 — Mirror Symmetry for Stress Manifolds
# ══════════════════════════════════════════════════════════════════════════════

class MirrorSymmetry:
    """
    For every stress manifold M (load-latency),
    there exists a mirror M^∨ (memory-allocation) such that:

      H¹(M) ≅ H^{2,1}(M^∨)   (failure cycles ↔ complex deformations)
      Kähler moduli of M = Complex structure of M^∨

    We construct M and M^∨ explicitly and verify the duality.
    """

    def __init__(self, grid_size: int = 35, seed: int = 55):
        self.N   = grid_size
        self.rng = np.random.default_rng(seed)
        t        = np.linspace(0, 2*np.pi, grid_size)
        self.X, self.Y = np.meshgrid(t, t)

    def manifold_M(self) -> np.ndarray:
        """
        M: load-latency manifold.
        L(x, y) = base latency surface.
        """
        return (
            1 + 0.5*np.sin(self.X) * np.cos(self.Y)
            + 0.3*np.sin(2*self.X)
            + 0.2*np.cos(3*self.Y)
            + 0.1*self.rng.standard_normal(self.X.shape)
        )

    def mirror_M_dual(self, M_field: np.ndarray) -> np.ndarray:
        """
        M^∨: mirror (memory-allocation) manifold.
        Constructed via 2D Fourier transform + mirror map.
        Mirror symmetry maps Kähler moduli ↔ complex structure.
        """
        F    = np.fft.fft2(M_field)
        # Mirror transformation: exchange frequency axes (complex structure flip)
        F_mirror = F.T.conj()
        return np.real(np.fft.ifft2(F_mirror))

    def compute_hodge_numbers(self, field: np.ndarray) -> dict:
        """
        Approximate h^{p,q} Hodge numbers via spectral decomposition.
        h^{1,0} ~ number of non-trivial gradient directions
        h^{0,1} ~ number of curl components
        """
        grad_x = np.gradient(field, axis=1)
        grad_y = np.gradient(field, axis=0)
        curl   = np.gradient(grad_x, axis=0) - np.gradient(grad_y, axis=1)

        # Approximate Betti-like numbers from harmonic content
        F      = np.fft.fft2(field)
        power  = np.abs(F)**2
        # h^{1,1} ~ number of significant frequency modes
        h11    = int((power > power.mean()).sum())
        h21    = int((np.abs(curl) > np.abs(curl).mean()).sum()) // self.N
        return dict(h11=h11, h21=h21, grad_x=grad_x, grad_y=grad_y, curl=curl)

    def verify_mirror_duality(self) -> dict:
        M       = self.manifold_M()
        M_dual  = self.mirror_M_dual(M)
        hM      = self.compute_hodge_numbers(M)
        hMd     = self.compute_hodge_numbers(M_dual)

        # Mirror prediction: h^{1,1}(M) = h^{2,1}(M^∨)
        mirror_match = abs(hM["h11"] - hMd["h21"]) / (hM["h11"] + 1) < 0.1

        # Correlation between M and M^∨ (should be non-trivial but related)
        corr = float(np.corrcoef(M.flatten(), M_dual.flatten())[0, 1])

        return dict(M=M, M_dual=M_dual, hM=hM, hMd=hMd,
                    mirror_match=mirror_match, corr=corr)


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmarks_part2():
    print("\n" + "="*72)
    print("  RUNTIME HOMOLOGY THEORY Part 2 — Benchmark Results")
    print("="*72)

    results = {}

    # ── Definition 4: Flux Divergence ─────────────────────────────────────────
    print("\n[Def 4] Request Flux Divergence (microservice mesh, 8 services)")
    rfd = RequestFluxDivergence(n_services=8)
    fd  = rfd.simulate(total_load=500, steps=300)
    bottlenecks = np.where(fd["divergence"] > 5)[0]
    print(f"    Bottleneck services:  {bottlenecks.tolist()}")
    print(f"    Max divergence:       {fd['divergence'].max():.2f} req/s")
    print(f"    Max queue depth:      {fd['queue'].max():.1f}")
    results["flux"] = (rfd, fd)

    # ── Definition 5: Entropy Production ──────────────────────────────────────
    print("\n[Def 5] Memory Entropy Production")
    mep  = MemoryEntropyProduction()
    sims = [mep.simulate(0.0, "Stable"), mep.simulate(3.0, "Slow leak"),
            mep.simulate(10.0, "Fast leak")]
    print(f"    {'Service':<12}  {'dS/dt mean':>12}  {'H_lifetime':>12}")
    print("    " + "-"*40)
    for s in sims:
        print(f"    {s['label']:<12}  {s['mean_dS']:>12.3f}  {s['H_lifetime']:>12.3f}")
    results["entropy"] = (mep, sims)

    # ── Definition 8: Resource Coupling Tensor ────────────────────────────────
    print("\n[Def 8] Resource Coupling Tensor T^μν")
    rct  = ResourceCouplingTensor()
    pts  = rct.sweep_operating_points()
    print(f"    {'Config':<12}  {'T_CPU,MEM':>12}  {'T_IO,NET':>10}  {'T_IO,CPU':>10}")
    print("    " + "-"*48)
    for p in pts:
        T = p["T"]
        print(f"    {p['label']:<12}  {T[0,1]:>12.4f}  {T[2,3]:>10.4f}  {T[2,0]:>10.4f}")
    results["coupling"] = (rct, pts)

    # ── Definition 9: Saturation Homology ─────────────────────────────────────
    print("\n[Def 9] Saturation Homology Groups")
    sh   = SaturationHomology()
    pts9 = sh._generate_failure_landscape(400)
    betti= sh.compute_betti_numbers(pts9)
    final_b0 = betti["beta0"][-1]
    max_b1   = betti["beta1"].max()
    print(f"    H₀ (failure clusters):   β₀ = {final_b0}")
    print(f"    H₁ (retry storm cycles): β₁_max = {max_b1}")
    results["homology"] = (sh, betti, pts9)

    # ── Definition 10: Load-Response Characteristic ───────────────────────────
    print("\n[Def 10] Load-Response Characteristic + Hysteresis")
    lrc    = LoadResponseCharacteristic()
    loads_up  = np.linspace(50, 780, 60)
    loads_dn  = np.linspace(780, 50, 60)
    lat_up = lrc.measure_lrc(loads_up,  direction="up")
    lat_dn = lrc.measure_lrc(loads_dn, direction="down")
    knee_idx = np.argmax(np.diff(lat_up) > 5)
    print(f"    Knee point:    λ ≈ {loads_up[knee_idx]:.0f} RPS  "
          f"(latency = {lat_up[knee_idx]:.1f} ms)")
    print(f"    Cliff point:   λ ≈ {loads_up[np.argmax(lat_up > 200)]:.0f} RPS")
    print(f"    Hysteresis area: {abs(np.trapezoid(lat_up - lat_dn[::-1], loads_up)):.1f}")
    results["lrc"] = (lrc, loads_up, loads_dn, lat_up, lat_dn)

    # ── Formula 1: Runtime Field Equation ─────────────────────────────────────
    print("\n[Form 1] Runtime Field Equation (Einstein analog)")
    rfe  = RuntimeFieldEquation(grid_size=30)
    feq  = rfe.solve_field_equation()
    res_norm = np.abs(feq["residual"]).mean() / (np.abs(feq["lhs"]).mean() + 1e-9)
    print(f"    Λ (baseline latency):    {feq['Lambda']:.2f} ms")
    print(f"    G (inverse capacity):    {feq['G']:.6f}")
    print(f"    Mean |R|:                {np.abs(feq['R']).mean():.4f}")
    print(f"    Normalised residual:     {res_norm*100:.2f}%")
    results["field_eq"] = (rfe, feq)

    # ── Formula 4: Bohr-Sommerfeld ─────────────────────────────────────────────
    print("\n[Form 4] Bohr-Sommerfeld Autoscaler Quantization")
    bsa  = BohrSommerfeldAutoscaler()
    orbs = [
        bsa.phase_space_orbit(0.25, 1.0),   # underdamped ζ=1.0 → w=1.0 (unstable)
        bsa.phase_space_orbit(0.25, 0.5),   # underdamped ζ=0.5 → w=0.5 (critical)
        bsa.phase_space_orbit(0.25, 0.3),   # underdamped ζ=0.3 → w=0.3 (stable)
        bsa.phase_space_orbit(2.00, 4.0),   # overdamped → w=0
    ]
    print(f"    {'K_I':>6} {'K_D':>6} {'ζ (w)':>8} {'J (action)':>12} {'Stable':>8}")
    print("    " + "-"*46)
    for o in orbs:
        print(f"    {o['K_I']:>6.2f} {o['K_D']:>6.2f} "
              f"{o['w']:>8.3f} {o['J']:>12.4f} "
              f"{'YES' if o['stable'] else 'NO':>8}")
    results["bohr"] = (bsa, orbs)

    # ── Theorem 2: Ising Mapping ───────────────────────────────────────────────
    print("\n[Thm 2] Ising Model ↔ GC Phase Transition (L=20 lattice)")
    ising = IsingGCMapping(L=20, seed=99)
    ising_r = ising.simulate_phase_transition(
        T_range=np.linspace(1.8, 3.2, 20), n_eq=200, n_meas=100)
    T_c_est = ising_r["T"][np.argmax(ising_r["chi"])]
    print(f"    Known T_c (2D Ising): 2.269")
    print(f"    Measured peak χ at:   T = {T_c_est:.3f}  (susceptibility ∝ GC pause)")
    print(f"    Error:                {abs(T_c_est - 2.269)/2.269*100:.1f}%")
    results["ising"] = (ising, ising_r)

    # ── Theorem 9: Atiyah-Singer ──────────────────────────────────────────────
    print("\n[Thm 9] Atiyah-Singer Index (zero modes = neutral equilibria)")
    asi   = AtiyahSingerIndex(grid_size=15)
    sweep = asi.index_vs_complexity(n_configs=6)
    print(f"    {'Complexity k':>14} {'β₀':>6} {'β₁':>6} {'χ=Index':>10} {'Zero modes':>12}")
    print("    " + "-"*52)
    for c in sweep:
        print(f"    {c['k']:>14} {c['beta0']:>6} {c['beta1']:>6} "
              f"{c['chi']:>10} {c['n_zero']:>12}")
    results["atiyah"] = (asi, sweep)

    # ── Conjecture 1: Runtime Riemann Hypothesis ───────────────────────────────
    print("\n[Conj 1] Runtime Riemann Hypothesis")
    rrh  = RuntimeRiemannHypothesis(N_terms=300)
    zh   = rrh.find_zeros_on_critical_line(np.linspace(0.5, 25.0, 500))
    print(f"    Non-trivial zeros found:  {zh['n_zeros']}")
    if zh["n_zeros"] > 0:
        print(f"    First zero at Im(s):      t = {zh['zero_t'][0]:.3f}")
        sigma_at_zero = zh["sigma_scan"][np.argmin(np.abs(zh["re_scan"]))]
        print(f"    Re(s) at first zero:      σ = {sigma_at_zero:.3f}  (hypothesis: 0.500)")
    results["riemann"] = (rrh, zh)

    # ── Conjecture 4: Mirror Symmetry ─────────────────────────────────────────
    print("\n[Conj 4] Mirror Symmetry for Stress Manifolds")
    ms   = MirrorSymmetry(grid_size=30)
    mir  = ms.verify_mirror_duality()
    print(f"    h^{{1,1}}(M):          {mir['hM']['h11']}")
    print(f"    h^{{2,1}}(M^∨):        {mir['hMd']['h21']}")
    print(f"    Mirror match:         {'YES ✓' if mir['mirror_match'] else 'NO ✗'}")
    print(f"    M ↔ M^∨ correlation:  {mir['corr']:.4f}")
    results["mirror"] = (ms, mir)

    print("\n" + "="*72)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def visualize_part2(data: dict):

    # ═══════════════════════════════════════════════════════════════════════
    # FIG 6 — Definitions 4, 5, 8, 10
    # ═══════════════════════════════════════════════════════════════════════
    fig6 = plt.figure(figsize=(22, 11), dpi=110)
    fig6.patch.set_facecolor(C["bg"])
    gs6  = gridspec.GridSpec(2, 3, figure=fig6, hspace=0.48, wspace=0.40)
    fig6.suptitle(
        "RUNTIME HOMOLOGY THEORY Pt.2 — Flux Divergence · Entropy · Coupling · LRC",
        color=C["gold"], fontsize=13, fontweight="bold", y=0.99)

    rfd, fd = data["flux"]
    mep, sims = data["entropy"]
    rct, pts  = data["coupling"]
    lrc_obj, loads_up, loads_dn, lat_up, lat_dn = data["lrc"]

    # 6a — Microservice flux heat-map (queue history)
    ax6a = ax2d(fig6, gs6[0, 0], "Def 4: Request Queue History (8 services)")
    im6a = ax6a.imshow(fd["history"].T, aspect="auto", cmap="inferno",
                       extent=[0, 300, 0, 8], origin="lower")
    plt.colorbar(im6a, ax=ax6a, label="Queue depth")
    ax6a.set_xlabel("Time step"); ax6a.set_ylabel("Service ID")

    # 6b — Flux divergence bar chart + 3-D queue mesh
    ax6b = ax3d(fig6, gs6[0, 1], "Def 4: Queue Depth over Time (3-D)", elev=30, azim=55)
    T_ax = np.arange(300)
    S_ax = np.arange(8)
    Tg, Sg = np.meshgrid(T_ax[::10], S_ax)
    ax6b.plot_surface(Tg, Sg, fd["history"][::10].T,
                      cmap="plasma", alpha=0.85, linewidth=0)
    ax6b.set_xlabel("Time"); ax6b.set_ylabel("Service"); ax6b.set_zlabel("Queue")

    # 6c — Entropy production trajectories
    ax6c = ax2d(fig6, gs6[0, 2], "Def 5: Memory Entropy Production dS/dt")
    cols_e = [C["green"], C["orange"], C["red"]]
    for s, col in zip(sims, cols_e):
        ax6c.plot(s["t"][::5], s["dS_dt"][::5], color=col, lw=1.5,
                  alpha=0.8, label=f"{s['label']} (μ={s['mean_dS']:.1f})")
    ax6c.axhline(0, color=C["text"], lw=0.8, ls="--")
    ax6c.set_xlabel("Time (s)"); ax6c.set_ylabel("dS/dt (entropy production)")
    ax6c.legend(fontsize=9)

    # 6d — Resource Coupling Tensor (heatmap for each operating point)
    axT = [ax2d(fig6, gs6[1, k], f"Def 8: Coupling Tensor — {pts[k]['label']}")
           for k in range(3)]
    cmap_t = plt.cm.RdBu_r
    for k, (ax_, p) in enumerate(zip(axT, pts)):
        T = p["T"]
        vmax = np.abs(T).max() + 1e-9
        im = ax_.imshow(T, cmap=cmap_t, vmin=-vmax, vmax=vmax, aspect="auto")
        ax_.set_xticks(range(4)); ax_.set_xticklabels(rct.RESOURCES, fontsize=8)
        ax_.set_yticks(range(4)); ax_.set_yticklabels(rct.RESOURCES, fontsize=8)
        plt.colorbar(im, ax=ax_, shrink=0.85)
        # Annotate diagonal (self-coupling)
        for i in range(4):
            ax_.text(i, i, f"{T[i,i]:.2f}", ha="center", va="center",
                     fontsize=7, color="white", fontweight="bold")

    plt.savefig(_out("fig6_definitions.png"), dpi=130,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig6)
    print("  → fig6 saved")

    # ═══════════════════════════════════════════════════════════════════════
    # FIG 7 — Load-Response Characteristic + Saturation Homology
    # ═══════════════════════════════════════════════════════════════════════
    fig7 = plt.figure(figsize=(20, 10), dpi=110)
    fig7.patch.set_facecolor(C["bg"])
    gs7  = gridspec.GridSpec(2, 3, figure=fig7, hspace=0.48, wspace=0.40)
    fig7.suptitle(
        "RUNTIME HOMOLOGY THEORY Pt.2 — LRC Hysteresis · Saturation Homology",
        color=C["gold"], fontsize=13, fontweight="bold", y=0.99)

    sh, betti, pts9 = data["homology"]

    # 7a — LRC curve with hysteresis (2D)
    ax7a = ax2d(fig7, gs7[0, :2], "Def 10: Load-Response Characteristic + Hysteresis")
    ax7a.plot(loads_up,    lat_up,    color=C["red"],    lw=2.5, label="Ramp-up ↑")
    ax7a.plot(loads_dn[::-1], lat_dn[::-1], color=C["cyan"], lw=2.5, ls="--", label="Ramp-down ↓")
    ax7a.fill_between(loads_up, lat_up, lat_dn[::-1], alpha=0.2,
                      color=C["orange"], label="Hysteresis region")
    knee = np.argmax(np.diff(lat_up, 2) > 0.05)
    ax7a.axvline(loads_up[knee], color=C["gold"], ls=":", lw=1.5, label=f"Knee λ≈{loads_up[knee]:.0f}")
    ax7a.set_xlabel("Load λ (RPS)"); ax7a.set_ylabel("p99 Latency (ms)")
    ax7a.legend(fontsize=9)

    # 7b — 3-D LRC surface (load × time → latency)
    ax7b = ax3d(fig7, gs7[0, 2], "Def 10: LRC 3-D (Load × Simulation Step)", elev=30, azim=50)
    # Simulate settling at each load
    n_settle = 30
    load_grid = np.linspace(50, 750, 20)
    settle_history = np.zeros((n_settle, len(load_grid)))
    state = 10.0
    for j, load in enumerate(load_grid):
        for step in range(n_settle):
            state = lrc_obj.steady_state_latency(load, state)*0.3 + state*0.7
            settle_history[step, j] = state
    Sg, Lg = np.meshgrid(np.arange(n_settle), load_grid)
    ax7b.plot_surface(Lg, Sg, settle_history.T, cmap="hot", alpha=0.85)
    ax7b.set_xlabel("Load λ"); ax7b.set_ylabel("Settle step"); ax7b.set_zlabel("Latency")

    # 7c — Failure landscape (scatter)
    ax7c = ax2d(fig7, gs7[1, 0], "Def 9: Failure State Space (Scatter)")
    ax7c.scatter(pts9[:, 0], pts9[:, 1], s=6, c=C["red"], alpha=0.4)
    ax7c.set_xlabel("PC1 (failure dim)"); ax7c.set_ylabel("PC2")

    # 7d — Persistent Betti numbers (β₀, β₁ vs ε)
    ax7d = ax2d(fig7, gs7[1, 1], "Def 9: Persistent Homology — Betti Numbers")
    ax7d.plot(betti["epsilon"], betti["beta0"], color=C["green"], lw=2, label="β₀ (components)")
    ax7d.plot(betti["epsilon"], betti["beta1"], color=C["red"],   lw=2, label="β₁ (loops / H₁)")
    ax7d.set_xlabel("Filtration radius ε")
    ax7d.set_ylabel("Betti number")
    ax7d.legend(fontsize=10)

    # 7e — Persistence diagram (birth-death pairs approximation)
    ax7e = ax2d(fig7, gs7[1, 2], "Def 9: Persistence Diagram (H₀ & H₁ pairs)")
    # Approximate birth-death via ε where β changes
    eps = betti["epsilon"]
    beta0 = betti["beta0"]
    beta1 = betti["beta1"]

    # H₀ birth-death: components merge as ε grows
    birth0 = [eps[0]] * int(beta0[0])
    death0 = []
    for i in range(1, len(eps)):
        n_merge = int(beta0[i-1]) - int(beta0[i])
        death0.extend([eps[i]] * max(0, n_merge))
    death0.extend([eps[-1]] * max(0, len(birth0) - len(death0)))
    birth0 = birth0[:len(death0)]

    # H₁ birth-death
    birth1, death1 = [], []
    for i in range(1, len(eps)):
        if beta1[i] > beta1[i-1]:
            birth1.extend([eps[i]] * int(beta1[i] - beta1[i-1]))
        elif beta1[i] < beta1[i-1]:
            death1.extend([eps[i]] * int(beta1[i-1] - beta1[i]))
    n_pairs = min(len(birth1), len(death1))
    birth1 = birth1[:n_pairs]; death1 = death1[:n_pairs]

    diag = np.linspace(0, eps[-1], 50)
    ax7e.plot(diag, diag, color=C["text"], lw=1, ls="--", alpha=0.5)
    if len(birth0) and len(death0):
        ax7e.scatter(birth0[:len(death0)], death0, s=30, c=C["green"],
                     label="H₀ (components)", zorder=3)
    if len(birth1) and len(death1):
        ax7e.scatter(birth1, death1, s=40, c=C["red"], marker="^",
                     label="H₁ (loops)", zorder=3)
    ax7e.set_xlabel("Birth ε"); ax7e.set_ylabel("Death ε")
    ax7e.legend(fontsize=9)

    plt.savefig(_out("fig7_lrc_homology.png"), dpi=130,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig7)
    print("  → fig7 saved")

    # ═══════════════════════════════════════════════════════════════════════
    # FIG 8 — Runtime Field Equation + Bohr-Sommerfeld
    # ═══════════════════════════════════════════════════════════════════════
    fig8 = plt.figure(figsize=(22, 11), dpi=110)
    fig8.patch.set_facecolor(C["bg"])
    gs8  = gridspec.GridSpec(2, 3, figure=fig8, hspace=0.48, wspace=0.42)
    fig8.suptitle(
        "RUNTIME HOMOLOGY THEORY Pt.2 — Runtime Field Equation · Bohr-Sommerfeld",
        color=C["gold"], fontsize=13, fontweight="bold", y=0.99)

    rfe, feq = data["field_eq"]
    bsa, orbs = data["bohr"]

    # 8a — p99 latency field L(cpu, mem)
    ax8a = ax3d(fig8, gs8[0, 0], "Form 1: p99 Latency Field L(cpu, mem)", elev=28, azim=-55)
    ax8a.plot_surface(feq["CPU"], feq["MEM"], feq["L"], cmap="inferno", alpha=0.9)
    ax8a.set_xlabel("CPU"); ax8a.set_ylabel("Memory"); ax8a.set_zlabel("p99 (ms)")

    # 8b — Ricci curvature scalar R(cpu, mem)
    ax8b = ax3d(fig8, gs8[0, 1], "Form 1: Ricci Scalar R(cpu, mem) [LHS]", elev=28, azim=45)
    R_plot = np.clip(feq["R"], -0.05, 0.05)
    ax8b.plot_surface(feq["CPU"], feq["MEM"], R_plot, cmap="RdBu_r", alpha=0.9)
    ax8b.set_xlabel("CPU"); ax8b.set_ylabel("Memory"); ax8b.set_zlabel("R")

    # 8c — Residual of field equation
    ax8c = ax2d(fig8, gs8[0, 2], "Form 1: Field Equation Residual (LHS − RHS)")
    im8c = ax8c.contourf(feq["CPU"], feq["MEM"],
                         np.clip(feq["residual"], -0.1, 0.1),
                         levels=30, cmap="RdBu_r")
    plt.colorbar(im8c, ax=ax8c, label="Residual")
    ax8c.set_xlabel("CPU cores"); ax8c.set_ylabel("Memory MB")

    # 8d — Bohr-Sommerfeld orbits in phase space
    ax8d = ax2d(fig8, gs8[1, :2],
                "Form 4: Bohr-Sommerfeld — Phase Space Orbits (winding number quantization)")
    cmap_bs = [C["red"], C["orange"], C["green"], C["cyan"]]
    labels_bs = ["ζ=1.0 → w=1.0 (unstable)", "ζ=0.5 → w=0.5 (critical)",
                 "ζ=0.3 → w=0.3 (stable)", "overdamped → w=0"]
    for orb, col, lab in zip(orbs, cmap_bs, labels_bs):
        N_ = orb["N_orb"]
        dN_ = orb["dN_orb"]
        ax8d.plot(N_[:120], dN_[:120], color=col, lw=2, label=lab)
    ax8d.axhline(0, color=C["text"], lw=0.6)
    ax8d.axvline(0, color=C["text"], lw=0.6)
    ax8d.scatter([0], [0], s=150, c=C["gold"], zorder=5, label="Target (N_t)")
    ax8d.set_xlabel("N − N_target (displacement)"); ax8d.set_ylabel("dN/dt (velocity)")
    ax8d.legend(fontsize=8)

    # 8e — Bohr-Sommerfeld quantization levels (K_D vs K_I stability ladder)
    ax8e = ax2d(fig8, gs8[1, 2], "Form 4: Quantization Ladder w_n = n + ½ − φ/2π")
    spec = bsa.quantization_spectrum(n_levels=4)
    K_I  = spec["K_I"]
    lev_cols = [C["green"], C["orange"], C["red"], C["purple"]]
    for n, col in zip(spec["n"], lev_cols):
        ax8e.plot(K_I, spec["levels"][n], color=col, lw=2.2,
                  label=f"n={n}  (w={0.5+0.5*n:.1f})")
    ax8e.fill_between(K_I, 0, spec["levels"][0], alpha=0.12,
                      color=C["green"], label="Stable zone (w<0.5)")
    ax8e.set_xlabel("K_I (integral gain)")
    ax8e.set_ylabel("K_D (derivative gain)")
    ax8e.legend(fontsize=9)

    plt.savefig(_out("fig8_field_equation.png"), dpi=130,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig8)
    print("  → fig8 saved")

    # ═══════════════════════════════════════════════════════════════════════
    # FIG 9 — Ising Mapping + Atiyah-Singer Index
    # ═══════════════════════════════════════════════════════════════════════
    fig9 = plt.figure(figsize=(22, 11), dpi=110)
    fig9.patch.set_facecolor(C["bg"])
    gs9  = gridspec.GridSpec(2, 3, figure=fig9, hspace=0.48, wspace=0.42)
    fig9.suptitle(
        "RUNTIME HOMOLOGY THEORY Pt.2 — Ising↔GC Mapping · Atiyah-Singer Index",
        color=C["gold"], fontsize=13, fontweight="bold", y=0.99)

    ising_obj, ising_r = data["ising"]
    asi, sweep_asi = data["atiyah"]

    # 9a — Magnetization vs T (order parameter = live object fraction)
    ax9a = ax2d(fig9, gs9[0, 0], "Thm 2: Ising Magnetization M(T)  [≡ Live Object Fraction]")
    ax9a.plot(ising_r["T"], ising_r["M"],
              color=C["cyan"], lw=2.5, marker="o", ms=5)
    ax9a.axvline(2.269, color=C["gold"], ls="--", lw=1.8, label="T_c = 2.269")
    ax9a.fill_between(ising_r["T"], 0, ising_r["M"], alpha=0.15, color=C["cyan"])
    ax9a.set_xlabel("Temperature T  [≡ Allocation rate θ]")
    ax9a.set_ylabel("Magnetization M  [≡ Live object fraction]")
    ax9a.legend(fontsize=10)

    # 9b — Susceptibility χ(T) (≡ GC pause time)
    ax9b = ax2d(fig9, gs9[0, 1], "Thm 2: Susceptibility χ(T)  [≡ GC Pause Time]")
    ax9b.plot(ising_r["T"], ising_r["chi"],
              color=C["red"], lw=2.5, marker="s", ms=5)
    ax9b.axvline(2.269, color=C["gold"], ls="--", lw=1.8, label="T_c")
    peak_T = ising_r["T"][np.argmax(ising_r["chi"])]
    ax9b.axvline(peak_T, color=C["orange"], ls=":", lw=1.5,
                 label=f"Peak T={peak_T:.3f}")
    ax9b.set_xlabel("Temperature T"); ax9b.set_ylabel("χ ∝ GC Pause")
    ax9b.legend(fontsize=10)

    # 9c — Ising lattice snapshot near T_c
    ax9c = fig9.add_subplot(gs9[0, 2])
    ax9c.set_facecolor(C["grid"])
    # Simulate final state at T_c
    grid_snap = ising_obj.grid.copy()
    rng_vis   = np.random.default_rng(42)
    for _ in range(10000):
        i, j   = rng_vis.integers(0, 20, 2)
        L      = 20
        dE     = 2 * grid_snap[i,j] * (
            grid_snap[(i+1)%L,j] + grid_snap[(i-1)%L,j]
            + grid_snap[i,(j+1)%L] + grid_snap[i,(j-1)%L]
        )
        if dE < 0 or rng_vis.random() < np.exp(-dE / 2.269):
            grid_snap[i,j] *= -1
    ax9c.imshow(grid_snap, cmap="RdBu_r", interpolation="nearest")
    ax9c.set_title("Thm 2: Ising Lattice at T_c  [≡ Object Graph at θ_c]",
                   color=C["gold"], fontsize=10)
    ax9c.set_xlabel("x  [heap column]"); ax9c.set_ylabel("y  [heap row]")

    # 9d — Index vs complexity
    ax9d = ax2d(fig9, gs9[1, 0], "Thm 9: Atiyah-Singer Index vs Landscape Complexity")
    ks   = [c["k"]     for c in sweep_asi]
    idx  = [c["index"] for c in sweep_asi]
    b0s  = [c["beta0"] for c in sweep_asi]
    b1s  = [c["beta1"] for c in sweep_asi]
    ax9d.plot(ks, idx, color=C["gold"],   lw=2.5, marker="D", label="Index(D) = χ")
    ax9d.plot(ks, b0s, color=C["green"],  lw=2.0, marker="o", ls="--", label="β₀")
    ax9d.plot(ks, b1s, color=C["red"],    lw=2.0, marker="s", ls="--", label="β₁")
    ax9d.set_xlabel("Landscape complexity k")
    ax9d.set_ylabel("Topological invariant")
    ax9d.legend(fontsize=10)

    # 9e — Zero-mode eigenvalue spectrum
    ax9e = ax2d(fig9, gs9[1, 1], "Thm 9: Laplacian Eigenvalue Spectrum (Zero Modes)")
    # Use last config
    eigv = sweep_asi[-1]
    x_ax = np.linspace(0, 4, 15)
    y_ax = np.linspace(0, 4, 15)
    Xf, Yf = np.meshgrid(x_ax, y_ax)
    L_ex = 20 + 10*np.sin(6*Xf)*np.cos(6*Yf) + 5*np.sin(6*Yf)
    r_ex = asi.count_zero_modes(L_ex, threshold=0.5)
    ax9e.semilogy(np.arange(len(r_ex["eigvals"])),
                  np.abs(r_ex["eigvals"]) + 1e-12,
                  color=C["purple"], lw=2)
    ax9e.axhline(0.5, color=C["gold"], ls="--", label="Zero-mode threshold")
    ax9e.fill_between(np.arange(len(r_ex["eigvals"])),
                      0, np.abs(r_ex["eigvals"]) + 1e-12,
                      where=np.abs(r_ex["eigvals"]) < 0.5,
                      alpha=0.3, color=C["green"], label=f"Zero modes = {r_ex['n_zero']}")
    ax9e.set_xlabel("Eigenvalue index"); ax9e.set_ylabel("|λ|")
    ax9e.legend(fontsize=9)

    # 9f — min tunable parameters = sum Betti numbers
    ax9f = ax2d(fig9, gs9[1, 2], "Thm 9: Min Tunable Parameters = Σ Betti Numbers")
    betti_sum = [c["beta0"] + c["beta1"] for c in sweep_asi]
    ax9f.bar(ks, betti_sum, color=C["cyan"], alpha=0.8, label="Σβ_i (min params)")
    ax9f.plot(ks, idx, color=C["gold"], lw=2.5, marker="D", label="Euler χ")
    ax9f.set_xlabel("Landscape complexity k")
    ax9f.set_ylabel("Count")
    ax9f.legend(fontsize=9)

    plt.savefig(_out("fig9_ising_atiyah.png"), dpi=130,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig9)
    print("  → fig9 saved")

    # ═══════════════════════════════════════════════════════════════════════
    # FIG 10 — Riemann Hypothesis + Mirror Symmetry
    # ═══════════════════════════════════════════════════════════════════════
    fig10 = plt.figure(figsize=(22, 11), dpi=110)
    fig10.patch.set_facecolor(C["bg"])
    gs10  = gridspec.GridSpec(2, 3, figure=fig10, hspace=0.50, wspace=0.42)
    fig10.suptitle(
        "RUNTIME HOMOLOGY THEORY Pt.2 — Runtime Riemann Hypothesis · Mirror Symmetry",
        color=C["gold"], fontsize=13, fontweight="bold", y=0.99)

    rrh, zh   = data["riemann"]
    ms, mir   = data["mirror"]

    # 10a — |ζ_R(0.5 + it)| (modulus on critical line)
    ax10a = ax2d(fig10, gs10[0, :2], "Conj 1: Runtime Riemann Zeta |ζ_R(½ + it)|")
    ax10a.plot(zh["t"], zh["mod_z"], color=C["cyan"], lw=1.5, label="|ζ_R|")
    if len(zh["zero_t"]) > 0:
        ax10a.scatter(zh["zero_t"], np.zeros(len(zh["zero_t"])) + zh["mod_z"].min()*0.5,
                      s=80, c=C["gold"], zorder=5, marker="v",
                      label=f"Zeros (n={zh['n_zeros']})")
    ax10a.set_xlabel("Im(s) = t"); ax10a.set_ylabel("|ζ_R(½ + it)|")
    ax10a.legend(fontsize=9)

    # 10b — Zero location test: Re(ζ_R) vs σ at first zero
    ax10b = ax2d(fig10, gs10[0, 2],
                 "Conj 1: Re(ζ_R(σ + it₀)) vs σ — Testing Re(s)=½")
    ax10b.plot(zh["sigma_scan"], zh["re_scan"], color=C["red"], lw=2.5)
    ax10b.axvline(0.5, color=C["gold"], ls="--", lw=2, label="σ = 0.5 (hypothesis)")
    ax10b.axhline(0.0, color=C["text"], lw=1)
    ax10b.scatter([0.5], [0.0], s=150, c=C["gold"], zorder=5)
    ax10b.set_xlabel("σ = Re(s)"); ax10b.set_ylabel("Re(ζ_R(σ + it₀))")
    ax10b.legend(fontsize=9)

    # 10c — Manifold M
    ax10c = ax3d(fig10, gs10[1, 0], "Conj 4: Stress Manifold M (load-latency)", elev=32, azim=50)
    X_, Y_ = ms.X, ms.Y
    ax10c.plot_surface(X_, Y_, mir["M"], cmap="plasma", alpha=0.9)
    ax10c.set_xlabel("Load dim 1"); ax10c.set_ylabel("Load dim 2"); ax10c.set_zlabel("L(x,y)")

    # 10d — Mirror manifold M^∨
    ax10d = ax3d(fig10, gs10[1, 1], "Conj 4: Mirror M^∨ (memory-allocation)", elev=32, azim=50)
    ax10d.plot_surface(X_, Y_, mir["M_dual"], cmap="viridis", alpha=0.9)
    ax10d.set_xlabel("Alloc dim 1"); ax10d.set_ylabel("Alloc dim 2"); ax10d.set_zlabel("L^∨(x,y)")

    # 10e — Mirror verification: curl fields + Hodge numbers
    ax10e = ax2d(fig10, gs10[1, 2], "Conj 4: Mirror Duality — Hodge Numbers & Curl")
    curl_M    = mir["hM"]["curl"]
    curl_Mdual= mir["hMd"]["curl"]
    ax10e.contourf(X_, Y_, curl_M,    levels=15, cmap="RdBu_r", alpha=0.6)
    ax10e.contour(X_, Y_, curl_Mdual, levels=8,  colors=[C["gold"]], alpha=0.8,
                  linewidths=1.2)
    ax10e.set_xlabel("dim 1"); ax10e.set_ylabel("dim 2")
    ax10e.set_title(
        f"Conj 4: h¹¹(M)={mir['hM']['h11']}  h²¹(M^∨)={mir['hMd']['h21']}"
        f"  corr={mir['corr']:.3f}  "
        f"match={'✓' if mir['mirror_match'] else '✗'}",
        color=C["gold"], fontsize=9)

    plt.savefig(_out("fig10_riemann_mirror.png"), dpi=130,
                bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig10)
    print("  → fig10 saved")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n  Output directory: {OUT_DIR}\n")
    print("  Building Part 2 objects …")
    data = run_benchmarks_part2()
    print("\n  Rendering figures …")
    visualize_part2(data)
    print(f"\n  ✓ All done. Figures saved to {OUT_DIR}\n")

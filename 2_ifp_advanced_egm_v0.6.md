---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
### Collaboration's notice


1. Improvements compared with V0.1:
   - Break the function ``egm_factory`` into two functions ``optimal_c`` and ``K``.
   - Convert it into ``MyST-md`` format by ``jupytext``.
   - Adjust the coefficient of $P$ and $\sigma$_vec and $\mu$-vec.
   - Converging time is **much shorter**.
   - Add assumption testings.
     - The assumption testing is on the money.
   - Change the loop structure.

2. Fault:
   - $\beta$ is not stochastic/state dependent.

3. Solution:
   - Simulate z_vec and P_z from the AR(1) $Z_t$ by using the function ``rouwenhorst`` in the ``QuantEcon.py``.
   - Set up appropriate expressions of stochastic processes, $\beta_t(z_t, ε_t)$, $R_t(z_t, ζ_t)$ and $Y_t(z_t, η_t)$.
<!-- #endregion -->

```python
import numpy as np
import matplotlib.pyplot as plt
from interpolation import interp
from numba import jit, njit, jitclass, prange, float64
from quantecon.optimize.root_finding import brentq
from scipy.linalg import eig, eigvals

%matplotlib inline
```

```python
ifp_data = [
    ('γ', float64),              # Utility parameter
    ('β', float64),
    ('P', float64[:, :]),        # Transition probs for z_t
    ('z_vec', float64[:]),       # Shock scale parameters for R_t, Y_t     
    ('a_grid', float64[:]),      # Grid over asset values (array)
    ('s_grid', float64[:]),
    ('ε_draws', float64[:]),
    ('η_draws', float64[:]),     # Draws of innovation η for MC (array)
    ('ζ_draws', float64[:])      # Draws of innovation ζ for MC (array)
]
```

```python hide-output=false
@jitclass(ifp_data)
class IFP:
    """
    A class that stores primitives for the income fluctuation 
    problem. 
    """
    def __init__(self,
                 γ=2.5,
                 β=0.96,
                 P=np.array([(0.7861, 0.2139), 
                             (0.2139, 0.7861)]),
                 z_vec=np.array((0.0316, 0.047)),
                 shock_draw_size=400,
                 grid_max=10,
                 grid_size=20):
        
        np.random.seed(1234)  # arbitrary seed

        self.γ = γ
        self.β = β
        self.P, self.z_vec = P, z_vec
        self.ε_draws = np.random.randn(shock_draw_size)
        self.η_draws = np.random.randn(shock_draw_size)
        self.ζ_draws = np.random.randn(shock_draw_size)
        self.a_grid = np.linspace(0, grid_max, grid_size)
        self.s_grid = np.copy(self.a_grid)
        
    # Marginal utility    
    def u_prime(self, c):
        return c ** (-self.γ)
    
    # Inverse utility
    def u_prime_inv(self, du):
        return du ** (-1 / self.γ)
    
    def R(self, z, ζ):
        return np.exp(self.z_vec[z] * ζ)
    
    def Y(self, z, η):
        return np.exp(self.z_vec[z] * η )
```

```python
ifp = IFP()
```

## Testing the assumptions

```python
def G_φ(Z_φ, P_φ):
    
    D_φ = np.diag(Z_φ)
    if D_φ.max() == np.inf:
        G_R = np.nan
    else:
        L_φ = np.matmul(P_φ, D_φ)
        G_φ = max(np.abs(eig(L_φ)[0])) # G_φ = r(L_φ)
    
    return G_φ
```

```python
G_βR = G_φ(ifp.β * np.exp(ifp.z_vec + (ifp.z_vec)**2 /2), ifp.P)
G_βR
```

## Implement the endogenous grid method

```python
@jit(nopython=True)
def optimal_c(s, z, σ_vec, ifp):
    
    u_prime, u_prime_inv = ifp.u_prime, ifp.u_prime_inv
    a_grid, s_grid = ifp.a_grid, ifp.s_grid
    η_draws, ζ_draws, ε_draws = ifp.η_draws, ifp.ζ_draws, ifp.ε_draws
    β, R, Y, P = ifp.β, ifp.R, ifp.Y, ifp.P
    n = len(ε_draws)
    
    c = lambda a, z: interp(a_grid, σ_vec[:, z], a)
        
    Ez = 0.0
    for z_hat in (0, 1):
        for i in range(n):
            R_hat = R(z_hat, ζ_draws[i])
            Y_hat = Y(z_hat, η_draws[i])
            Ez += β * R_hat * u_prime(c(R_hat * s + Y_hat, z_hat)) * P[z, z_hat]
                
    Ez = Ez / (n ** 3)
    return u_prime_inv(Ez)
    
```

```python
@jit(nopython=True)
def K(c, ifp):
    
    a_grid, s_grid = ifp.a_grid, ifp.s_grid
    
    c_new = np.empty_like(c)
    for z in (0, 1):
        for i in range(len(s_grid)):
            s = s_grid[i]
            
            c_new[i, z] = optimal_c(s, z, c, ifp)
            
        a_new = c_new[:, z] + s_grid
        c_new[:, z] = interp(a_new, c_new[:, z], a_grid)
        
        c_new[:, z] = np.minimum(c_new[:, z], a_grid)
        
    return c_new
```

```python
def solve_model(ifp,
                K,
                tol=1e-10,
                max_iter=1e3,
                verbose=True,
                print_skip=4):

    """
    Solves for the optimal policy using operator K

    * ifp is an instance of ConsumerProblem
    * K is an operator that updates consumption policy
    """

    # Initial guess of c_vec = consume all assets
    n = len(ifp.a_grid)
    c = np.empty((n, 2))
    for z in 0, 1:
        c[:, z] = ifp.a_grid

    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        c_new = K(c, ifp)

        error = np.max(np.abs(c - c_new))
        i += 1

        if verbose and i % print_skip == 0:
            print(f"Error at iteration {i} is {error}.")

        c[:, :] = c_new

    if i == max_iter:
        print("Failed to converge!")

    if verbose and i < max_iter:
        print(f"\nConverged in {i} iterations.")

    return c
```

```python
%%time
c = solve_model(ifp, K)
```

```python
for z in [0, 1]:
    plt.plot(ifp.a_grid, c[:, z], label=f"$z$ = {z}")

plt.legend()
plt.show()
```

```python

```

```python

```

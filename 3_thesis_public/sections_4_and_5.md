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

```python
import solve_model as sm
import inequality_measure_plot as imp
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


# %matplotlib inline
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
```

# 1 The experiment with aggregate shocks

## 1.1 Step 1-2

```python
%%time
ifp6 = sm.IFP(a_y=0.5, b_y=1.0, c_y=1.0,
              a_r=0.2270572, b_r=0.1, c_r=0.03433458, 
              grid_max=10000,
             grid_size=1000)  

k = len(ifp6.s_grid)
n = len(ifp6.P)

σ_init = np.empty((k, n))
for z in range(n):
    σ_init[:, z] = ifp6.s_grid
a_init = np.copy(σ_init)

a_good_seq6, a_bad_seq6, σ_good_seq6, σ_bad_seq6, a_star6, σ_star6 = sm.solve_model_time_iter(ifp6, a_init, σ_init, print_skip=5)
print("beta, beta R, R_mean and sR are", ifp6.β, ifp6.βR, ifp6.R_mean, ifp6.sR)
```

## 1.2 Step 3-6: figure 4

```python
%%time
ginis6, tail_index_hill6, tail_index_gabaix6, z_seq_new6, dists6 = imp.dynamic_inequality(ifp6, a_star6, σ_star6,
                                                                                          num_households=2000,
                                                                                          burn_in_length=50,
                                                                                          sim_length=2000,
                                                                                          c=1,
                                                                                         z=0)

imp.plot_dynamic_inequality(z_seq_new6, ginis6, tail_index_hill6, tail_index_gabaix6,
                           path='/Users/shuhu/Desktop/1_papers_w:john/3_thesis/figures/fig52.png')

print('Respectively, the mean, variance and autocorrelation of ginis are', 
      np.mean(ginis6), 
      np.var(ginis6), 
      np.mean(sm.estimated_autocorrelation(ginis6)))
```

# 2 The experiment without aggregate shocks

## 2.1 Step 1-2

```python
%%time
ifp7 = sm.IFP(a_y=0.5, b_y=0.0, c_y=1.0,
              a_r=0.2270572, b_r=0.0, c_r=0.03433458, 
              grid_max=10000,
             grid_size=1000)  # grid_size=xx means that we have xx households in our economy

k = len(ifp7.s_grid)
n = len(ifp7.P)

σ_init = np.empty((k, n))
for z in range(n):
    σ_init[:, z] = ifp7.s_grid
a_init = np.copy(σ_init)

a_good_seq7, a_bad_seq7, σ_good_seq7, σ_bad_seq7, a_star7, σ_star7 = sm.solve_model_time_iter(ifp7, a_init, σ_init, print_skip=5)
print("beta, beta R, R_mean and sR are", ifp7.β, ifp7.βR, ifp7.R_mean, ifp7.sR)
```

## 2.2 Step 3-6: figure 3

```python
%%time
ginis7, tail_index_hill7, tail_index_gabaix7, z_seq_new7, dists7 = imp.dynamic_inequality(ifp7, a_star7, σ_star7,
                                                                                          num_households=2000,
                                                                                          burn_in_length=50,
                                                                                          sim_length=2000,
                                                                                          c=1,
                                                                                         z=0)

imp.plot_dynamic_inequality(z_seq_new7, ginis7, tail_index_hill7, tail_index_gabaix7,
                           path='/Users/shuhu/Desktop/1_papers_w:john/3_thesis/figures/fig53.png')

print('Respectively, the mean, variance and autocorrelation of ginis are', 
      np.mean(ginis7), 
      np.var(ginis7), 
      np.mean(sm.estimated_autocorrelation(ginis7)))
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

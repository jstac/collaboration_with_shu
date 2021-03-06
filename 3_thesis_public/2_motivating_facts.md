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

# 2_motivating_facts_v5


With the dataset ``SCF+`` and methodology from [Kuhn, Schularick and Steins (2020)](https://www.journals.uchicago.edu/doi/10.1086/708815), we estimate the dynamic Gini coefficients on wealth and income for the U.S. households from 1950 to 2016 for three wealth-level samples:
- all wealth levels
- bottom 95%
- bottom 90%

The dataset is saved as ``SCF_plus.dta``, and we do the estimation using Stata, please see [here](https://github.com/jstac/collaboration_with_shu/tree/master/3_thesis_public/generate_ginis).

In this section, first, we tabularly and graphically present the dynamic wealth and income Gini coefficients and measures their mean, variance and autocorrelation coefficients, which would become targets for our calibration and quantitative experiments. Then we briefly discuss those observations and stylized facts, in accordingly with some questions of interest.

```python
# with some imports

import solve_model as sm
import inequality_measure_plot as imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## 2.1 wealth and income ginis in the U.S.

```python
df = pd.read_csv('https://raw.githubusercontent.com/jstac/inequality_dynamics/master/thesis/generate_ginis/ginis_all_b99_b90.csv?token=AM3OYILBL35YLAK6KGPKPU264ZDSW')
```

```python
df
```

### 2.1.1 window time 1: from 1950 to 2016

```python
year1, income_ginis_all, wealth_ginis_all = imp.generate_ginis(df)
year2, income_ginis_99, wealth_ginis_99 = imp.generate_ginis(df, income_ginis_xx='gini_tinc_B99', wealth_ginis_xx='gini_ffanw_B99')
year3, income_ginis_90, wealth_ginis_90 = imp.generate_ginis(df, income_ginis_xx='gini_tinc_B90', wealth_ginis_xx='gini_ffanw_B90')

year = [year1, year2, year3]
income_ginis = [income_ginis_all, income_ginis_99, income_ginis_90]
wealth_ginis = [wealth_ginis_all, wealth_ginis_99, wealth_ginis_90]
```

```python
imp.plot_generating_ginis(year, income_ginis, wealth_ginis)
```

```python
imp.measure_mean_var_autocorr(income_ginis_all)
```

```python
imp.measure_mean_var_autocorr(wealth_ginis_all, ginis_type='wealth ginis-all')
```

### 2.1.2 window time 2: from 1965 to 1983

Let's zoom in the pictures to see the stylized facts about income and wealth Gini coefficients during the so-called [oil crisis](https://en.wikipedia.org/wiki/1973_oil_crisis) period

```python
df2 = df[5:10]
df2
```

```python
year4, income_ginis_all2, wealth_ginis_all2 = imp.generate_ginis(df2)
year5, income_ginis_992, wealth_ginis_992 = imp.generate_ginis(df2, income_ginis_xx='gini_tinc_B99', wealth_ginis_xx='gini_ffanw_B99')
year6, income_ginis_902, wealth_ginis_902 = imp.generate_ginis(df2, income_ginis_xx='gini_tinc_B90', wealth_ginis_xx='gini_ffanw_B90')

year2 = [year4, year5, year6]
income_ginis2 = [income_ginis_all2, income_ginis_992, income_ginis_902]
wealth_ginis2 = [wealth_ginis_all2, wealth_ginis_992, wealth_ginis_902]
```

```python
imp.plot_generating_ginis(year2, income_ginis2, wealth_ginis2)
```

```python
imp.measure_mean_var_autocorr(income_ginis_all2)
```

```python
imp.measure_mean_var_autocorr(wealth_ginis_all2, ginis_type='wealth ginis-all')
```

```python

```

### 2.1.3 window time 3: from 2000 to 2016

Let's move out of the window time 2 and zoom in the big pictures again to see the stylized facts about income and wealth Gini coefficients during the so-called
[global financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%932008) period.

```python
df3 = df[14:]
df3
```

```python
year7, income_ginis_all3, wealth_ginis_all3 = imp.generate_ginis(df3)
year8, income_ginis_993, wealth_ginis_993 = imp.generate_ginis(df3, income_ginis_xx='gini_tinc_B99', wealth_ginis_xx='gini_ffanw_B99')
year9, income_ginis_903, wealth_ginis_903 = imp.generate_ginis(df3, income_ginis_xx='gini_tinc_B90', wealth_ginis_xx='gini_ffanw_B90')

year3 = [year7, year8, year9]
income_ginis3 = [income_ginis_all3, income_ginis_993, income_ginis_903]
wealth_ginis3 = [wealth_ginis_all3, wealth_ginis_993, wealth_ginis_903]
```

```python
imp.plot_generating_ginis(year3, income_ginis3, wealth_ginis3)
```

```python
imp.measure_mean_var_autocorr(income_ginis_all3)
```

```python
imp.measure_mean_var_autocorr(wealth_ginis_all3, ginis_type='wealth ginis-all')
```

## 2.2 discussions

Please see my thesis (update soon).

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

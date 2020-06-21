# SCF+ dataset and generating ginis

## 1 usage on files

1. The ``SCF+`` dataset is saved in ``SCF_plus.dta``.
   - All monetary variables are in 2016 dollars, deflated using the CPI of the same year (note that income variables refer to the previous year).
   - There are 5 implicates per observation due to multiple imputation. The implicates are indicated by the variable "impnum".
   - For detailed variables, please see the following table.
2. The codes to produce gini coefficients are collected in``generate_gini.do``.
3. Income and wealth Gini coefficients across $3$ wealth pools from $1950$ to $2016$ are stored in the table ``ginis_all_b99_b90``.
   - $3$ wealth pools are whole sample, bottom $99\%$ and bottom $90\%$.
   - The table is saved with two file extensions, but their contents are the same.

## 2 overview over ``SCF+`` variables

|  variable name  |                            label                             |
| :-------------: | :----------------------------------------------------------: |
|     adults      |                       number of adults                       |
|      ageh       |                         age of head                          |
|    agehgroup    |                          age group                           |
|     blackh      |                    whether head is black                     |
|       bnd       |                            bonds                             |
|     ccdebt      |                       credit card debt                       |
|      cerde      |                   certificates of deposit                    |
|    children     |                      number of children                      |
|    collegeh     |       whether head has attained at least some college        |
|       CPI       |                     consumer price index                     |
|     ffaass      |                         total assets                         |
|     ffabus      |                       business wealth                        |
|     ffaequ      |               equity and other managed assets                |
|     ffafin      | financial assets (ffaequ, liqcer, bnd, mfun, ofin, life, pen) |
|     ffanfin     |   non-financial assets (ffabus, house, oest, vehi, onfin)    |
|      ffanw      |            net wealth (ffafin + ffanfin - tdebt)             |
|   ffanwgroups   |                        wealth groups                         |
|      hdebt      |          housing debt on owner-occupied real estate          |
|     hhequiv     |                    OECD equivalence scale                    |
|   highsample    |           indicator for high-income sample in 1983           |
|      house      |                     asset value of house                     |
| housing_rent_yd |       housing rental yield from Macrohistory Database        |
|       id        |                         household id                         |
|     impnum      |                 imputation implicate number                  |
|     inccap      |                        capital income                        |
|    inctrans     |                       transfer income                        |
|      incws      |       income from wages, salaries and self-employment        |
|     incwsse     |                income from wages and salaries                |
|      life       |                    life insurance assets                     |
|       liq       |                        liquid assets                         |
|     liqcer      |          liquid assets and certificates of deposit           |
|      mfun       |                         mutual funds                         |
| moneymarketacc  |                    money market accounts                     |
|      oest       |               other real estate (net position)               |
|    oestdebt     |                    other real estate debt                    |
|      ofin       |                    other financial assets                    |
|      onfin      |                  other non-financial assets                  |
|     othdebt     |                          other debt                          |
|       PCE       |           personal consumption expenditures index            |
|      pdebt      |                        personal debt                         |
|       pen       |                           pensions                           |
|     prepaid     |                        prepaid cards                         |
|      raceh      |                         race of head                         |
|     savbnd      |                        savings bonds                         |
|      tdebt      | total household debt (excluding other real estate debt, i.e. hdebt + pdebt) |
|      tinc       |       total household income, excluding capital gains        |
|   tincgroups    |                        income groups                         |
|      vehi       |                           vehicles                           |
|       wgt       |                      unadjusted weight                       |
|    wgtI95W95    |                        survey weight                         |
|      year       |                             year                             |
|    yearmerge    |                        3-year window                         |



#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binom, chi2_contingency
from statsmodels.stats.proportion import (
    proportions_chisquare,
    proportions_ztest,
    test_proportions_2indep,
)

n = 1000

rows = []
for i in range(100):
    x1 = binom(n, 0.01).rvs()
    x2 = binom(n, 0.02).rvs()

    stat, pvalue = test_proportions_2indep(
        x1, n, x2, n, method="score", correction=True
    )
    rows.append({"stat": stat, "pvalue": pvalue, "method": "score", "i": i})

    # print("score test statsmodels", pvalue)

    stat, pvalue, table = proportions_chisquare([x1, x2], [n, n])
    rows.append({"stat": stat, "pvalue": pvalue, "method": "chi2", "i": i})

    # print("chi2 statsmodels:", pvalue)

    table = np.array([[x1, n - x1], [x2, n - x2]])
    stat, pvalue, dof, table = chi2_contingency(table, correction=False)
    rows.append({"stat": stat, "pvalue": pvalue, "method": "chi2 scipy", "i": i})

    # print("chi2 scipy", pvalue)

    stat, pvalue = proportions_ztest([x1, x2], [n, n])
    rows.append({"stat": stat, "pvalue": pvalue, "method": "ztest", "i": i})

    # print("ztest statsmodels", pvalue)

results = pd.DataFrame(rows)

#%%

results = results.sort_values("pvalue")

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sns.lineplot(data=results, x="i", y="pvalue", hue="method", ax=ax)

#%%
diffs = (
    results.query('method == "score"').set_index("i")["pvalue"]
    - results.query('method == "chi2"').set_index("i")["pvalue"]
)

sns.histplot(diffs)


# Mutual Information Estimator

This library estimates **mutual information (MI)** between two variables using two methods:

1. **KSG estimator (K-Nearest Neighbors)** for continuous variables.
2. **Binning / Histogram-based estimator**.

---

## Usage

```python
import numpy as np
from mutual_info_estimator import mutual_information_ksg, mutual_information_binning

# Example data
X = np.random.rand(1000, 1)
Y = np.sin(2 * np.pi * X) + 0.1 * np.random.randn(1000, 1)

# KSG estimator
mi_knn = mutual_information_ksg(X, Y, k=5)
print("Mutual Information (KSG):", mi_knn)

# Binning estimator
mi_bin = mutual_information_binning(X, Y, bins=20)
print("Mutual Information (Binning):", mi_bin)
```

---

## Notes

- **KSG Estimator**: Good for continuous and high-dimensional variables. More accurate than binning, but computationally heavier.
- **Binning Estimator**: Easy to use, works for discrete or discretized continuous variables. Accuracy depends on number of bins.
- Add `np.atleast_2d` if your input is 1D to avoid shape issues.

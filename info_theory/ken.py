import numpy as np, matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)

plt.figure(figsize=(6,4))
plt.plot(x, y, marker='o', markersize=3, linewidth=1)  # line + point markers
plt.grid(True); plt.tight_layout(); plt.show()
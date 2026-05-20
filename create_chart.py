import matplotlib.pyplot as plt
import numpy as np

# Create a simple line chart
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 24, 36, 18, 42])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, marker='o', linewidth=2, markersize=8, label='Sales Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Sales ($1000s)')
ax.set_title('Monthly Sales Performance')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('chart.png', dpi=100, bbox_inches='tight')
print("Chart created: chart.png")

import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(12.8, 4.8))

# celebaqh
bs =     [45.33, 45.33, 45.33, 45.33, 45.33]
hard_y = [42.92, 44.14, 44.75, 45.02, 45.23]
easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =      [0.1,   0.5,   1,     2,     5]
plt.subplot(1, 2, 1)
plt.grid(b=True)
plt.xticks(x)
plt.plot(x, bs, marker='v', markersize=8, label='without weighting')
plt.plot(x, hard_y, marker='*', markersize=8, label='hard weighting')
plt.plot(x, easy_y, marker='o', label='easy weighting')
plt.legend(loc='upper right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Temperature $\\beta$", fontsize=12)
plt.ylabel("FID Score", fontsize=12)
plt.title("Horse $\\rightarrow$ Zebra", fontsize=16)
# plt.savefig('unpaired_graph.png', bbox_inches='tight')


# ade20k
bs =     [33.42, 33.42, 33.42, 33.42, 33.42]
hard_y = [35.83, 34.64, 34.05, 33.82, 33.46]
easy_y = [32.47, 32.73, 32.92, 33.15, 33.41]
x =      [0.1,   0.5,   1,     2,     5]
plt.subplot(1, 2, 2)
plt.grid(b=True)
plt.xticks(x)
plt.plot(x, bs, marker='v', markersize=8, label='without weighting')
plt.plot(x, hard_y, marker='*', markersize=8, label='hard weighting')
plt.plot(x, easy_y, marker='o', label='easy weighting')
plt.legend(loc='upper right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Temperature $\\beta$", fontsize=12)
plt.ylabel("FID Score", fontsize=12)
plt.title("ADE20K (Semantic)", fontsize=16)
plt.savefig('unpaired_graph.pdf', bbox_inches='tight')
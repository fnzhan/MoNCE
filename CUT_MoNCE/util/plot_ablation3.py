import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(25.6, 4.8))

# hosrse.
PatchNCE =  [45.33, 45.33, 45.33, 45.33, 45.33]
WeightNCE = [42.92, 44.14, 44.75, 45.02, 45.23]
MoNCE =     [41.86, 43.47, 44.40, 44.89, 45.23]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [0.1,   0.5,   1,     2,     5]
plt.subplot(1, 4, 1)
plt.grid(b=True)
plt.xticks(x)
plt.plot(x, PatchNCE, marker='v', markersize=8, label='PatchNCE')
plt.plot(x, WeightNCE, marker='*', markersize=8, label='WeightNCE')
plt.plot(x, MoNCE, marker='o', label='MoNCE')
plt.legend(loc='center right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Temperature $\\beta$", fontsize=12)
plt.ylabel("FID Score", fontsize=12)
plt.title("Horse $\\rightarrow$ Zebra", fontsize=16)
# plt.savefig('unpaired_graph.png', bbox_inches='tight')


# ade20k
PatchNCE =  [33.42, 33.42, 33.42, 33.42, 33.42]
WeightNCE = [32.47, 32.73, 32.92, 33.15, 33.41]
MoNCE =     [31.62, 32.14, 32.62, 32.99, 33.38]
x =         [0.1,   0.5,   1,     2,     5]
plt.subplot(1, 4, 2)
plt.grid(b=True)
plt.xticks(x)
plt.plot(x, PatchNCE, marker='v', markersize=8, label='PatchNCE')
plt.plot(x, WeightNCE, marker='*', markersize=8, label='WeightNCE')
plt.plot(x, MoNCE, marker='o', label='MoNCE')
plt.legend(loc='center right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Temperature $\\beta$", fontsize=12)
plt.ylabel("FID Score", fontsize=12)
plt.title("ADE20K (Semantic)", fontsize=16)
# plt.savefig('unpaired_graph.pdf', bbox_inches='tight')










PatchNCE =  [46.13, 45.83, 45.33, 44.43, 43.53]
WeightNCE = [44.79, 43.99, 42.92, 42.62, 42.23]
MoNCE =     [43.86, 42.97, 41.86, 41.61, 41.45]
x =         [0.1,   0.5,   1,     2,    5]
plt.subplot(1, 4, 3)
plt.grid(b=True)
plt.xticks(x)
plt.plot(x, PatchNCE, marker='v', markersize=8, label='PatchNCE')
plt.plot(x, WeightNCE, marker='*', markersize=8, label='WeightNCE')
plt.plot(x, MoNCE, marker='o', label='MoNCE')
plt.legend(loc='upper right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Negative Term Weight Q", fontsize=12)
plt.ylabel("FID Score", fontsize=12)
plt.title("Horse $\\rightarrow$ Zebra", fontsize=16)


PatchNCE =  [32.54, 32.88, 33.42, 34.10, 34.72]
WeightNCE = [31.83, 32.03, 32.47, 33.42, 34.29]
MoNCE =     [31.31, 31.43, 31.56, 32.62, 33.86]
x =         [0.1,   0.5,   1,     2,     5]
plt.subplot(1, 4, 4)
plt.grid(b=True)
plt.xticks(x)
plt.plot(x, PatchNCE, marker='v', markersize=8, label='PatchNCE')
plt.plot(x, WeightNCE, marker='*', markersize=8, label='WeightNCE')
plt.plot(x, MoNCE, marker='o', label='MoNCE')
plt.legend(loc='lower right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Negative Term Weight Q", fontsize=12)
plt.ylabel("FID Score", fontsize=12)
plt.title("ADE20K (Semantic)", fontsize=16)
plt.savefig('im_ablation.pdf', bbox_inches='tight')
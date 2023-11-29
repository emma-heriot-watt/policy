import numpy as np
import matplotlib.pyplot as plt

ablation_percentages = np.array([0, 0.25, 0.5, 0.75, 1.0])

# the 2nd and the second to last are swapped
vision_aug_results = np.array([27.15, 34.72, 35.24, 36.64, 36.81])
cdf_aug_results = np.array([34.09, 34.89, 36.61, 36.81, 36.81])
human_results = np.array([19.17, 19.17, 19.17, 19.17, 19.17])
y_ticks = [17, 20, 23, 26, 29, 32, 35, 38]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ablation_percentages, human_results, "-.g", label="DTC")
ax.plot(ablation_percentages, vision_aug_results, "-bs", label="Visual Aug")

ax2 = ax.twinx()
ax2.plot(ablation_percentages, cdf_aug_results, "--ro", label="CDF Aug")
# fig.legend(loc="upper right")

# ax.set_xlabel("Pr")
ax.set_yticks(y_ticks)
ax2.set_yticks(y_ticks)
ax.set_ylabel(r"Vision Augmentations")
ax2.set_ylabel(r"CDF Augmentations")

ax.grid()

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.425), fancybox=True, ncol=3)
plt.xticks(ablation_percentages, ablation_percentages)
# plt.show()
plt.title("Performance curves when ablating augmentations")
ax.set_xlabel("Proportion of train instances")
plt.savefig("human.pdf", transparent=True)

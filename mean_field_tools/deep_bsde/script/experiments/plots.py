import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, layout="constrained", figsize=(9, 6))

plt.savefig("./poster_plots/placeholder_measure_flow.png")


fig, axs = plt.subplots(1, 3, layout="constrained", figsize=(9, 3))
axs[0].set_xlabel(r"$(X^i_t - X_t)$")
axs[1].set_xlabel(r"$(Y^i_t - Y_t)$")
axs[2].set_xlabel(r"$(m^i_t - m_t)$")

plt.savefig("./poster_plots/placeholder_convergence.png")

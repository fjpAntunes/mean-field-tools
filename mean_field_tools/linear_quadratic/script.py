import numpy as np
import matplotlib.pyplot as plt
from ode_solver import LinearQuadraticMFG

arg_list = [
    "A",
    "barA",
    "B",
    "C",
    "Q",
    "barQ",
    "S",
    "QT",
    "barQT",
    "ST",
    "x0",
    "sigma0",
    "nu",
]

arg_dict = {name: 1 for name in arg_list}

# arg_dict["barQT"] = 2.45
arg_dict["nu"] = 0.5
arg_dict["sigma0"] = 0.2

time_domain = np.linspace(0, 1, 101)

lq_mfg = LinearQuadraticMFG(time_domain, **arg_dict)

initial_z = np.ones(time_domain.shape)
initial_r = initial_z

if __name__ == "__main__":
    values = lq_mfg.solve_nash_equilibrium(
        initial_z, initial_r, damping=0, num_iterations=100
    )
    fig, axs = plt.subplots(4, figsize=(6, 18))

    labels = ["z", "p", "r", "s"]

    for i, ax in enumerate(axs):
        ax.plot(time_domain, values[i], "b", label=f"{labels[i]}(t)")
        ax.legend(loc="best")
        ax.set_xlabel("t")
        ax.grid()

    plt.savefig("./linear_quadratic/output/states.png", format="png")

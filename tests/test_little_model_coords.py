import numpy as np
import matplotlib.pyplot as plt


def get_grid(Nx, Ny, dx, dy, x0, y0):

    x1 = x0 + Nx*dx
    y1 = y0 + Ny*dy

    # x = np.linspace(x0, x1, Nx)
    # y = np.linspace(y0, y1, Ny)

    x = np.arange(x0, x1, dx)
    y = np.arange(y0, y1, dy)

    xv, yv = np.meshgrid(x, y)

    return xv, yv


def make_ca1_pyrs_coords(X0, Y0, dx, dy, Npyrs_sim_x, Npyrs_sim_y, Nbottom_gens_x, Nbottom_gens_y, Nleft_gens_x, Nleft_gens_y):

    x0 = X0 + dx * Nleft_gens_x
    y0 = Y0 + dy * Nbottom_gens_y



    xv_sim, yv_sim = get_grid(Npyrs_sim_x, Npyrs_sim_y, dx, dy, x0, y0)


    xv_gens_bottom, yv_gens_bottom = get_grid(Nbottom_gens_x, Nbottom_gens_y, dx, dy, X0, Y0)

    yv_gens_top = yv_gens_bottom + dy*(Npyrs_sim_y + Nbottom_gens_y)
    xv_gens_top = np.copy(xv_gens_bottom)

    x0_left = X0
    y_left = y0
    xv_gens_left, yv_gens_left = get_grid(Nleft_gens_x, Nleft_gens_y, dx, dy, x0_left, y_left)

    yv_gens_right = np.copy(yv_gens_left)
    xv_gens_right = xv_gens_left + dx*(Nleft_gens_x + Npyrs_sim_x)

    xv_gens = np.concatenate( [xv_gens_left.ravel(), xv_gens_top.ravel(),xv_gens_right.ravel(), xv_gens_bottom.ravel()]  )
    yv_gens = np.concatenate([yv_gens_left.ravel(), yv_gens_top.ravel(), yv_gens_right.ravel(), yv_gens_bottom.ravel()])

    xv_sim = xv_sim.ravel()
    yv_sim = yv_sim.ravel()

    return xv_sim, yv_sim, xv_gens, yv_gens



X0 = 0
Y0 = 0
dx = 15
dy = 15
Nbottom_gens_x = 13
Nbottom_gens_y = 3

Npyrs_sim_x = 7
Npyrs_sim_y = 7

Nleft_gens_x = 3
Nleft_gens_y = Npyrs_sim_y

xv_sim, yv_sim, xv_gens, yv_gens = make_ca1_pyrs_coords(X0, Y0, dx, dy, Npyrs_sim_x, Npyrs_sim_y, Nbottom_gens_x, Nbottom_gens_y, Nleft_gens_x, Nleft_gens_y)

plt.scatter(xv_sim, yv_sim, marker='^', c='blue', s=50)
plt.scatter(xv_gens, yv_gens, marker='^', c='green', s=50)
# plt.scatter(xv_gens_bottom, yv_gens_bottom, marker='^', c='green', s=50)
# plt.scatter(xv_gens_top, yv_gens_top, marker='^', c='green', s=50)
# plt.scatter(xv_gens_left, yv_gens_left, marker='^', c='green', s=50)
# plt.scatter(xv_gens_right, yv_gens_right, marker='^', c='green', s=50)


plt.show()
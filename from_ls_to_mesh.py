import numpy as np
import matplotlib.pyplot as plt
import os
import dolfin as df
import time
from pymedit import (
    P1Function,
    square,
    mmg2d,
    trunc,
)


def create_standard_mesh(phi, hmax=0.05, plot_mesh=False, return_times=False):
    """Generation of a mesh over a domain defined by a level-set function.

    Args:
        phi (array): array of values of the level-set function
        hmax (float, optional): maximal size of cell. Defaults to 0.05.
        plot_mesh (bool, optional): plot the resulting mesh or not. Defaults to False.

    Returns:
        mesh: a FEniCS mesh.
    """
    t0 = time.time()
    n = np.shape(phi)[0]
    M = square(n - 1, n - 1)
    t1 = time.time()
    construct_background_mesh = t1 - t0
    M.debug = 4  # For debugging and mmg3d output

    # Setting a P1 level set function
    phi = phi.flatten("F")
    t0 = time.time()
    phiP1 = P1Function(M, phi)
    t1 = time.time()
    interp_time = t1 - t0
    # Remesh according to the level set
    t0 = time.time()
    newM = mmg2d(
        M,
        hmax=hmax / 1.43,
        hmin=hmax / 2,
        hgrad=None,
        sol=phiP1,
        ls=True,
        verb=0,
    )
    t1 = time.time()
    remesh_time = t1 - t0
    # Trunc the negative subdomain of the level set
    t0 = time.time()
    Mf = trunc(newM, 3)
    t1 = time.time()
    trunc_mesh = t1 - t0
    Mf.save("Thf.mesh")  # Saving in binary format
    command = "meshio convert Thf.mesh Thf.xml"
    t0 = time.time()
    os.system(command)
    t1 = time.time()
    conversion_time = t1 - t0
    t0 = time.time()
    mesh = df.Mesh("Thf.xml")
    t1 = time.time()
    fenics_read_time = t1 - t0
    if plot_mesh:
        plt.figure()
        df.plot(mesh, color="purple")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    if return_times:
        return mesh, [
            construct_background_mesh,
            interp_time,
            remesh_time,
            trunc_mesh,
            conversion_time,
            fenics_read_time,
        ]
    else:
        return mesh


if __name__ == "__main__":

    n = 512
    x, y = np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n)
    XX, YY = np.meshgrid(x, y)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    def phi(x, y):
        return -1.0 + ((x - 0.5) / 0.38) ** 2 + ((y - 0.5) / 0.23) ** 2

    phi_np = phi(XXYY[0, :], XXYY[1, :]).reshape(n, n)

    mesh, mesh_times = create_standard_mesh(
        phi=phi_np, hmax=0.05, plot_mesh=True, return_times=True
    )
    bd_points = df.BoundaryMesh(mesh, "exterior", True).coordinates()
    vals_phi = phi(bd_points[:, 0], bd_points[:, 1])

    error_mean = np.mean(np.absolute(vals_phi) ** 2) ** 0.5
    error_min = np.min(np.absolute(vals_phi))
    error_max = np.max(np.absolute(vals_phi))
    print(f"{error_mean=:3e} {error_min=:3e} {error_max=:3e}")
    plt.figure(figsize=(6, 6))
    plt.contourf(x, y, phi_np, levels=50, cmap="viridis")
    df.plot(mesh)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(
        f"Boundary errors : MSE ={error_mean:3e} \nmin = {error_min:3e} max = {error_max:3e}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig("from_ls_to_mesh.pdf")
    plt.show()

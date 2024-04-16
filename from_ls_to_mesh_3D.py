import numpy as np
import matplotlib.pyplot as plt
import os
import dolfin as df
import time
from pymedit import (
    P1Function3D,
    mmg3d,
    trunc3DMesh,
    cube,
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
    M = cube(n - 1, n - 1, n - 1)
    t1 = time.time()
    construct_background_mesh = t1 - t0
    M.debug = 4  # For debugging and mmg3d output
    print("ok")
    # Setting a P1 level set function
    phi = phi.transpose(1, 0, 2).flatten()
    t0 = time.time()
    phiP1 = P1Function3D(M, phi)
    t1 = time.time()
    print("ok")
    interp_time = t1 - t0
    # Remesh according to the level set
    t0 = time.time()
    newM = mmg3d(
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
    print("ok")
    # Trunc the negative subdomain of the level set
    t0 = time.time()
    Mf = trunc3DMesh(newM, 3)
    t1 = time.time()
    trunc_mesh = t1 - t0
    print("ok")
    Mf.save("Thf.mesh")
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
        import vedo
        import vedo.dolfin as vdf

        vdf.plot(mesh, color="purple")
        vedo.close()

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

    n = 128
    xyz = np.linspace(0.0, 1.0, n)
    XX, YY, ZZ = np.meshgrid(xyz, xyz, xyz)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    ZZ = np.reshape(ZZ, [-1])
    XXYYZZ = np.stack([XX, YY, ZZ])

    def phi(x, y, z):
        return (
            -1.0
            + ((x - 0.5) / 0.23) ** 2
            + ((y - 0.5) / 0.28) ** 2
            + ((z - 0.5) / 0.35) ** 2
        )

    phi_np = phi(XXYYZZ[0, :], XXYYZZ[1, :], XXYYZZ[2, :]).reshape(n, n, n)

    mesh, mesh_times = create_standard_mesh(
        phi=phi_np, hmax=0.06, plot_mesh=True, return_times=True
    )
    bd_points = df.BoundaryMesh(mesh, "exterior", True).coordinates()
    vals_phi = phi(bd_points[:, 0], bd_points[:, 1], bd_points[:, 2])

    error_mean = np.mean(np.absolute(vals_phi) ** 2) ** 0.5
    error_min = np.min(np.absolute(vals_phi))
    error_max = np.max(np.absolute(vals_phi))
    print(f"{error_mean=:1e} {error_min=:1e} {error_max=:1e}")
    import vedo
    import vedo.dolfin as vdf

    my_plt = vdf.plot(mesh, color="purple", interactive=False, axes=0, size=(600, 600))
    text = f"Boundary errors : MSE = {error_mean:1e} \nmin = {error_min:1e}  max = {error_max:1e}"
    formula = (
        vedo.Text3D(text, c="black", s=0.015)
        .pos(0.08, 0.5, 0.45)
        .rotate_z(45, around=(0.08, 0.5, 0.45))
    )
    actors = my_plt.actors[:]
    actors.append(formula)

    vedo.show(
        actors,
        formula,
        axes=0,
        interactive=True,
        size=[600, 600],
    )

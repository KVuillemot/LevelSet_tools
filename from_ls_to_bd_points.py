import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
import time


def build_exact_boundary(
    phi, alpha=0.01, max_iterations=25, xmin=0, xmax=1, ymin=0, ymax=1
):
    """
    Creation of a precise set of boundary points starting from an array of level-set function values.

    """
    x = np.linspace(xmin, xmax, phi.shape[0])
    y = np.linspace(ymin, ymax, phi.shape[1])
    start = time.time()
    spline_phi = RectBivariateSpline(x, y, phi.T, kx=3, ky=3)
    end = time.time()
    create_spline = end - start
    XX, YY = np.meshgrid(x, y)
    XX = np.reshape(XX, [-1])
    YY = np.reshape(YY, [-1])
    XXYY = np.stack([XX, YY])

    start = time.time()
    interpolated_value = spline_phi(XXYY[0, :], XXYY[1, :], grid=False)
    indice_init_boundary_point = np.argmin(np.absolute(interpolated_value))
    initial_boundary_point = XXYY[:, indice_init_boundary_point]
    end = time.time()
    find_init_point = end - start
    ordered_points = []
    point = initial_boundary_point
    N = 0
    start = time.time()

    while (
        np.abs(spline_phi(point[0], point[1], grid=False)) > 1e-14
        and N < max_iterations * 20
    ):
        grad_phi = np.array(
            [
                spline_phi(point[0], point[1], dx=1, dy=0, grid=False),
                spline_phi(point[0], point[1], dx=0, dy=1, grid=False),
            ]
        )
        phi_i = spline_phi(point[0], point[1], grid=False)
        point = point - phi_i * grad_phi / (np.linalg.norm(grad_phi) ** 2 + 1e-5)
        N += 1
    ordered_points.append(point)
    end = time.time()
    first_ordered_point = end - start

    start = time.time()
    while (
        np.linalg.norm(ordered_points[0] - ordered_points[-1], ord=2) == 0.0
        or np.linalg.norm(ordered_points[0] - ordered_points[-1], ord=2) > alpha
        or len(ordered_points) < 10
    ):
        origin = ordered_points[-1]
        grad_phi = np.array(
            [
                spline_phi(origin[0], origin[1], dx=1, dy=0, grid=False),
                spline_phi(origin[0], origin[1], dx=0, dy=1, grid=False),
            ]
        )
        ortho = np.array([-grad_phi[1], grad_phi[0]])
        p_1 = origin + alpha * ortho / (grad_phi[0] ** 2 + grad_phi[1] ** 2 + 1e-5)

        point = p_1
        N = 0
        while (
            np.abs(spline_phi(point[0], point[1], grid=False)) > 1e-14
            and N < max_iterations
        ):
            grad_phi = np.array(
                [
                    spline_phi(point[0], point[1], dx=1, dy=0, grid=False),
                    spline_phi(point[0], point[1], dx=0, dy=1, grid=False),
                ]
            )
            phi_i = spline_phi(point[0], point[1], grid=False)
            point = point - phi_i * grad_phi / (
                grad_phi[0] ** 2 + grad_phi[1] ** 2 + 1e-5
            )

            N += 1
        if len(ordered_points) > 10:
            v1 = np.linalg.norm(ordered_points[0] - ordered_points[-1], ord=2)
            v2 = np.linalg.norm(ordered_points[-1] - point, ord=2)
            v3 = np.linalg.norm(ordered_points[0] - point, ord=2)
            if (v2 >= max(v1, v3)) and (v1 < alpha or v3 < alpha):
                break

        ordered_points.append(point)
    ordered_points = np.array(ordered_points)
    end = time.time()
    main_loop = end - start
    print(f"{create_spline=:.3f}")
    print(f"{find_init_point=:.3f}")
    print(f"{first_ordered_point=:.3f}")
    print(f"{main_loop=:.3f}")
    return ordered_points


if __name__ == "__main__":

    test_case = 2  # 1 : circle 2 : star

    if test_case == 1:
        n = 256
        xmin, xmax, ymin, ymax = 0, 1, 0, 1
        x, y = np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n)
        XX, YY = np.meshgrid(x, y)
        XX = np.reshape(XX, [-1])
        YY = np.reshape(YY, [-1])
        XXYY = np.stack([XX, YY])

        def phi(x, y):
            return -(0.3**2) + (x - 0.5) ** 2 + (y - 0.5) ** 2

        phi_np = phi(XXYY[0, :], XXYY[1, :]).reshape(n, n)
        boundary_points = build_exact_boundary(
            phi_np, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
        )

        vals_phi = phi(boundary_points[:, 0], boundary_points[:, 1])

        error_mean = np.mean(np.absolute(vals_phi) ** 2) ** 0.5
        error_min = np.min(np.absolute(vals_phi))
        error_max = np.max(np.absolute(vals_phi))
        print(f"{error_mean=:3e} {error_min=:3e} {error_max=:3e}")

        plt.figure(figsize=(10, 10))
        plt.contourf(x, x, phi_np, levels=50, cmap="viridis")
        plt.plot(
            boundary_points[:, 0],
            boundary_points[:, 1],
            "+",
            color="r",
            label="reconstructed boundary points",
        )
        exact_boundary_points = np.array(
            [
                [0.5 + 0.3 * np.cos(t), 0.5 + 0.3 * np.sin(t)]
                for t in np.linspace(0, 2.0 * np.pi, 5000)
            ]
        )
        plt.plot(
            exact_boundary_points[:, 0],
            exact_boundary_points[:, 1],
            "k-",
            label="exact",
        )
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.show()

    else:
        n = 256
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
        x, y = np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n)
        XX, YY = np.meshgrid(x, y)
        XX = np.reshape(XX, [-1])
        YY = np.reshape(YY, [-1])
        XXYY = np.stack([XX, YY])

        def phi(x, y):
            return x**2 + y**2 - (0.6 + 0.2 * np.sin(6 * np.arctan(y / x)))

        phi_np = phi(XXYY[0, :], XXYY[1, :]).reshape(n, n)
        boundary_points = build_exact_boundary(
            phi_np, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
        )

        vals_phi = phi(boundary_points[:, 0], boundary_points[:, 1])

        error_mean = np.mean(np.absolute(vals_phi) ** 2) ** 0.5
        error_min = np.min(np.absolute(vals_phi))
        error_max = np.max(np.absolute(vals_phi))
        print(f"{error_mean=:3e} {error_min=:3e} {error_max=:3e}")

        plt.figure(figsize=(10, 10))
        plt.contourf(x, x, phi_np, levels=50, cmap="viridis")
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], "-+", color="k")
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.show()

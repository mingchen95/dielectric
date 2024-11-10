import numpy as np
import MDAnalysis as mda
import cupy as cp  # 使用 cupy 代替 numpy
from cupy.fft import fftn  # 使用 GPU 加速 FFT
import argparse
import time

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute structure factor from molecular dynamics trajectory.")
    parser.add_argument('-f', '--trajectory', type=str, required=True, help="Path to the XTC trajectory file.")
    parser.add_argument('-s', '--topology', type=str, required=True, help="Path to the TPR topology file.")
    parser.add_argument('-dr', '--grid_spacing', type=float, required=True, help="Grid spacing (in Å).")
    parser.add_argument('-b', '--begin', type=float, default=None, help="First frame to read (in ps).")
    parser.add_argument('-e', '--end', type=float, default=None, help="Last frame to read (in ps).")
    parser.add_argument('-kr_max', '--kr_max', type=float, default=25.0, help="Maximum kr value (in Å⁻¹) for calculation.")
    parser.add_argument('-dk', '--dk', type=float, default=0.02, help="Interval for kr output (in Å⁻¹).")
    parser.add_argument('-o', '--order', type=int, default=6, help="Order of Lagrangian interpolation (supported: 3, 4, 5, 6).")
    parser.add_argument('-pme', '--pme', type=int, default=0, help="pme=0, direct calculation; pme=1, lagrangian_interpolation with PME.")
    return parser.parse_args()

def initialize_simulation(topology_file, trajectory_file, grid_spacing):
    """Initialize the MDAnalysis universe and calculate grid parameters."""
    u = mda.Universe(topology_file, trajectory_file)
    box_size = u.trajectory.ts.dimensions[:3]
    nx, ny, nz = (cp.ceil(cp.array(box_size) / grid_spacing)).astype(int)
    voltot = cp.prod(cp.array(box_size))
    return u, cp.array(box_size), nx, ny, nz, voltot

def precompute_kr(nx, ny, nz, grid_spacing):
    """Precompute the kr values for the FFT grid."""
    kx = 2 * cp.pi * cp.fft.fftfreq(nx, d=grid_spacing)
    ky = 2 * cp.pi * cp.fft.fftfreq(ny, d=grid_spacing)
    kz = 2 * cp.pi * cp.fft.fftfreq(nz, d=grid_spacing)
    kx, ky, kz = cp.meshgrid(kx, ky, kz, indexing='ij')
    kr = cp.sqrt(kx**2 + ky**2 + kz**2)
    return kr

def lagrangian_interpolation(order, dx):
    """Lagrangian interpolation for different orders."""
    n = len(dx)
    if order == 3:
        coeffs = cp.zeros((3, n))
        coeffs[0] = (1 - 3 * dx + 3 * dx ** 2 - dx ** 3) / 6
        coeffs[1] = (4 - 6 * dx ** 2 + 3 * dx ** 3) / 6
        coeffs[2] = (1 + 3 * dx + 3 * dx ** 2 - 3 * dx ** 3) / 6
    elif order == 4:
        coeffs = cp.zeros((4, n))
        coeffs[0] = (-dx ** 3 + 3 * dx ** 2 - 3 * dx + 1) / 6
        coeffs[1] = (3 * dx ** 3 - 6 * dx ** 2 + 4) / 6
        coeffs[2] = (-3 * dx ** 3 + 3 * dx ** 2 + 3 * dx + 1) / 6
        coeffs[3] = (dx ** 3) / 6
    elif order == 5:
        coeffs = cp.zeros((5, n))
        coeffs[0] = (-1 + 5 * dx - 10 * dx ** 2 + 10 * dx ** 3 - 5 * dx ** 4 + dx ** 5) / 120
        coeffs[1] = (19 - 65 * dx + 95 * dx ** 2 - 55 * dx ** 3 + 10 * dx ** 4) / 120
        coeffs[2] = (46 - 45 * dx ** 2 + 15 * dx ** 4) / 60
        coeffs[3] = (19 + 65 * dx + 95 * dx ** 2 + 55 * dx ** 3 + 10 * dx ** 4) / 120
        coeffs[4] = (-1 - 5 * dx - 10 * dx ** 2 - 10 * dx ** 3 - 5 * dx ** 4 - dx ** 5) / 120
    elif order == 6:
        coeffs = cp.zeros((6, n))
        coeffs[0] = (1 - 10 * dx + 40 * dx ** 2 - 80 * dx ** 3 + 80 * dx ** 4 - 32 * dx ** 5) / 3840
        coeffs[1] = (237 - 750 * dx + 840 * dx ** 2 - 240 * dx ** 3 - 240 * dx ** 4 + 160 * dx ** 5) / 3840
        coeffs[2] = (841 - 770 * dx - 440 * dx ** 2 + 560 * dx ** 3 + 80 * dx ** 4 - 160 * dx ** 5) / 1920
        coeffs[3] = (841 + 770 * dx - 440 * dx ** 2 - 560 * dx ** 3 + 80 * dx ** 4 + 160 * dx ** 5) / 1920
        coeffs[4] = (237 + 750 * dx + 840 * dx ** 2 + 240 * dx ** 3 - 240 * dx ** 4 - 160 * dx ** 5) / 3840
        coeffs[5] = (1 + 10 * dx + 40 * dx ** 2 + 80 * dx ** 3 + 80 * dx ** 4 + 32 * dx ** 5) / 3840
    else:
        raise ValueError(f"Unsupported interpolation order {order}! Only 3, 4, 5, and 6 are supported.")
    return coeffs

def charge_assignment(u, grid_spacing, nx, ny, nz, order, pme):
    all_atoms = u.atoms
    positions = cp.array(all_atoms.positions)
    charges = cp.array(all_atoms.charges)
    nat = len(charges)  # 原子总数
    charge_mesh = cp.zeros((nx, ny, nz))  # 初始化网格上的电荷分布

    # 归一化坐标并计算最近网格点
    uu = positions / grid_spacing
    midpoint = cp.floor(uu) + 0.5
    dx = uu - midpoint
    gross_index = cp.floor(uu).astype(int) - order // 2 + 1

    if pme == 0:
        # 直接将电荷分配到网格点上
        grid_indices = cp.round(positions / grid_spacing).astype(int)
        for iat in range(nat):
            if charges[iat] != 0:
                grid_size = cp.array([nx, ny, nz])
                idx = (grid_indices[iat] % grid_size)
                charge_mesh[idx[0], idx[1], idx[2]] += charges[iat]
    elif pme == 1:
        # 使用 PME 算法进行插值
        for iat in range(nat):
            if charges[iat] != 0:
                # 计算插值系数
                lagrange_x = lagrangian_interpolation(order, dx[iat, 0])
                lagrange_y = lagrangian_interpolation(order, dx[iat, 1])
                lagrange_z = lagrangian_interpolation(order, dx[iat, 2])

                # 计算电荷对网格点的贡献
                charge_mesh_contrib = charges[iat] * (
                    cp.outer(lagrange_x, cp.outer(lagrange_y, lagrange_z)).reshape(order, order, order)
                )

                # 使用 meshgrid 和广播来避免嵌套循环
                idx_x = (gross_index[iat, 0] + cp.arange(order)) % nx
                idx_y = (gross_index[iat, 1] + cp.arange(order)) % ny
                idx_z = (gross_index[iat, 2] + cp.arange(order)) % nz

                # 创建索引网格
                ix, iy, iz = cp.meshgrid(idx_x, idx_y, idx_z, indexing='ij')

                # 将电荷贡献加到 charge_mesh 上
                cp.add.at(charge_mesh, (ix, iy, iz), charge_mesh_contrib)

    return charge_mesh

def compute_frame(ts, nx, ny, nz, grid_spacing, kr, order, u, pme):
    """Compute the charge density and its FFT for a single frame."""
    start_time = time.time()
    charge_mesh = charge_assignment(u, grid_spacing, nx, ny, nz, order, pme)

    rho_k = fftn(charge_mesh)
    rho_k_squared = cp.abs(rho_k) ** 2
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"compute_frame_in_blocks execution time: {elapsed_time:.4f} seconds")
    return rho_k, rho_k_squared

def main():
    args = parse_arguments()

    # Convert begin and end to integers
    if args.begin is not None:
        args.begin = int(args.begin)
    if args.end is not None:
        args.end = int(args.end) if args.end != -1 else None

    u, box_size, nx, ny, nz, voltot = initialize_simulation(args.topology, args.trajectory, args.grid_spacing)
    kr = precompute_kr(nx, ny, nz, args.grid_spacing)

    nx, ny, nz = int(nx), int(ny), int(nz)
    rho_k_sum = cp.zeros((nx, ny, nz), dtype=cp.complex128)
    rho_k_squared_sum = cp.zeros((nx, ny, nz), dtype=cp.complex128)
    frame_count = 0
	
    L = cp.min(box_size)
    k_min = 2 * cp.pi / L

    for ts in u.trajectory[args.begin:args.end]:
        if ts.frame % 500 == 0:
            print(f"Processing frame {ts.frame}")

        rho_k_frame, rho_k_squared_frame = compute_frame(ts, nx, ny, nz, args.grid_spacing, kr, args.order, u, args.pme)

        rho_k_sum += rho_k_frame
        rho_k_squared_sum += rho_k_squared_frame
        frame_count += 1

    average_rho_k = cp.abs(rho_k_sum / frame_count) / cp.sqrt(voltot)
    average_rho_k_squared = cp.abs(rho_k_squared_sum / frame_count) / voltot

    #output_file_name = "epsilon_largeK.txt"
    #dk_values = args.dk
    dk_values = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    kr_max = args.kr_max
    for dk_local in dk_values:
        output_file_name = f"epsilon_largeK_{dk_local:.2f}.txt"
        with open(output_file_name, 'w') as file:
            kr_range = cp.arange(k_min, kr_max, dk_local)
            for k in kr_range:
                indices = cp.isclose(kr, k,  atol=dk_local/2)
                if cp.any(indices):
                    avg_rho_k = cp.mean(average_rho_k[indices.get()])
                    avg_square = cp.mean(average_rho_k_squared[indices.get()])
                    avg_Sk = avg_square / k / k
                    avg_chi = 4 * cp.pi * avg_Sk * 557
                    ave_eps = 1 / (1 - avg_chi)
                    file.write(f"{k:.2f} {avg_rho_k:.6e} {avg_square:.6e} {avg_Sk:.6e} {avg_chi:.6e} {ave_eps:.6e}\n")


if __name__ == "__main__":
    main()



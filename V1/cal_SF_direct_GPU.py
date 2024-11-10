import MDAnalysis as mda
import cupy as cp
import numpy as np
import argparse
import gc
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute structure factor from molecular dynamics trajectory.")
    parser.add_argument('-f', '--trajectory', type=str, required=True, help="Path to the XTC trajectory file.")
    parser.add_argument('-s', '--topology', type=str, required=True, help="Path to the TPR topology file.")
    parser.add_argument('-dr', '--grid_spacing', type=float, required=True, help="Grid spacing (in Å).")
    parser.add_argument('-b', '--begin', type=float, default=None, help="First frame to read (in ps).")
    parser.add_argument('-e', '--end', type=float, default=None, help="Last frame to read (in ps).")
    parser.add_argument('-kr_max', '--kr_max', type=float, default=25.0, help="Maximum kr value (in Å⁻¹).")
    parser.add_argument('-dk', '--dk', type=float, default=0.02, help="Interval for kr output (in Å⁻¹).")
    parser.add_argument('-n', '--block_size', type=int, default=10, help="Number of atoms to process in each block.")
    parser.add_argument('-N', '--expand_factor', type=int, default=1, help="Expansion factor for the simulation box.")
    parser.add_argument('--molecule', type=str, help="Molecule resname to select (e.g., 'SOL' for water or 'NA CL' for ions).")    
    return parser.parse_args()


def initialize_simulation(topology_file, trajectory_file, grid_spacing):
    u = mda.Universe(topology_file, trajectory_file)
    box_size = u.trajectory.ts.dimensions[:3]
    nx, ny, nz = (np.ceil(box_size / grid_spacing)).astype(int)
    voltot = box_size[0] * box_size[1] * box_size[2]  # volume of the expanded simulation box
    return u, cp.array(box_size), nx, ny, nz, voltot


def precompute_filtered_kr(nx, ny, nz, grid_spacing, K_max):
    """Precompute filtered kx, ky, kz values for the FFT grid on the GPU based on K_max."""
    # Compute original kx, ky, kz values
    kx = 2 * cp.pi * cp.fft.fftfreq(nx, d=grid_spacing)
    ky = 2 * cp.pi * cp.fft.fftfreq(ny, d=grid_spacing)
    kz = 2 * cp.pi * cp.fft.fftfreq(nz, d=grid_spacing)

    # Filter frequencies where each component is >= 0 and < K_max
    kx_filtered = kx[(kx >= 0) & (kx < K_max)]
    ky_filtered = ky[(ky >= 0) & (ky < K_max)]
    kz_filtered = kz[(kz >= 0) & (kz < K_max)]

    # Create new meshgrid based on filtered frequencies
    kx_new, ky_new, kz_new = cp.meshgrid(kx_filtered, ky_filtered, kz_filtered, indexing='ij')
    kr_new = cp.sqrt(kx_new ** 2 + ky_new ** 2 + kz_new ** 2)

    return kx_new, ky_new, kz_new, kr_new

def select_molecules(u, molecule_resname):
    """根据传入的 resname 选择原子，如果 resname 为 None 则选择整个系统。"""
    if molecule_resname is None:
        return u.atoms  # 选择整个系统的所有原子
    else:
        return u.select_atoms(f"resname {molecule_resname}")


def compute_frame_in_blocks(ts, kx, ky, kz, u, block_size, selected_atoms):
    start_time = time.time()
    all_atoms = u.atoms
    #positions = cp.array(all_atoms.positions)  # Transfer positions to GPU
    #charges = cp.array(all_atoms.charges)  # Transfer charges to GPU    
    positions = cp.array(selected_atoms.positions)
    charges = cp.array(selected_atoms.charges)
    atom_indices = selected_atoms.indices  # 获取选定原子的索引
    print(f"Frame {ts.frame}: Selected atom indices: {atom_indices}")
    num_atoms = positions.shape[0]

    # Initialize rho_k for the entire k-space
    rho_k = cp.zeros(kx.shape, dtype=cp.complex128)

    # Process atoms in blocks to reduce memory footprint
    for start in range(0, num_atoms, block_size):
        end = min(start + block_size, num_atoms)

        # Get the positions and charges for the current block of atoms
        pos_block = positions[start:end]
        charges_block = charges[start:end]

        pos_x, pos_y, pos_z = pos_block[:, 0], pos_block[:, 1], pos_block[:, 2]

        # Compute the phase factors for this block
        phase_factors_block = cp.exp(
            -1j * (kx[cp.newaxis, :, :, :] * pos_x[:, cp.newaxis, cp.newaxis, cp.newaxis] +
                   ky[cp.newaxis, :, :, :] * pos_y[:, cp.newaxis, cp.newaxis, cp.newaxis] +
                   kz[cp.newaxis, :, :, :] * pos_z[:, cp.newaxis, cp.newaxis, cp.newaxis]))

        # Sum the contributions from the current block of atoms
        rho_k += cp.sum(charges_block[:, cp.newaxis, cp.newaxis, cp.newaxis] * phase_factors_block, axis=0)

    # Compute the squared magnitude (|rho_k|^2)
    rho_k_squared = cp.abs(rho_k) ** 2

    end_time = time.time()
    # Calculate and print execution time
    elapsed_time = end_time - start_time
    # print(f"compute_frame_in_blocks execution time: {elapsed_time:.4f} seconds")

    return rho_k, rho_k_squared


def main():
    args = parse_arguments()

    if args.begin is not None:
        args.begin = int(args.begin)
    if args.end is not None:
        args.end = int(args.end) if args.end != -1 else None

    u, box_size, nx, ny, nz, voltot = initialize_simulation(args.topology, args.trajectory, args.grid_spacing)
    kx, ky, kz, kr = precompute_filtered_kr(nx, ny, nz, args.grid_spacing, args.kr_max)
    selected_atoms = select_molecules(u, args.molecule)

    L = cp.min(box_size)
    k_min = 2 * cp.pi / L

    rho_k_sum = cp.zeros(kx.shape, dtype=cp.complex128)
    rho_k_square_sum = cp.zeros(kx.shape, dtype=cp.float64)
    frame_count = 0

    for ts in u.trajectory[args.begin:args.end]:
        if ts.frame % 100 == 0:
            print(f"Processing frame {ts.frame}")

        # Compute rho_k and rho_k_squared for the current frame, in blocks
        rho_k_frame, rho_k_squared_frame = compute_frame_in_blocks(ts, kx, ky, kz, u, args.block_size, selected_atoms)
        rho_k_sum += rho_k_frame
        rho_k_square_sum += rho_k_squared_frame
        frame_count += 1

        # Release memory for rho_k_frame and result_frame after processing each frame
        del rho_k_frame
        del rho_k_squared_frame
        gc.collect()  # Force garbage collection

    # Normalize and calculate average (transfer data to CPU for final operations if necessary)
    average_rho_k = cp.abs(rho_k_sum / frame_count) / cp.sqrt(voltot)
    average_rho_squared = rho_k_square_sum / frame_count / voltot
    average_Sk1 = average_rho_squared / (kr ** 2)
    average_Sk1[kr == 0] = 0
    average_chi1 = 4 * cp.pi * 557 * average_Sk1
    average_epsilon1 = 1 / (1 - average_chi1)

    # Transfer results back to NumPy for output if necessary
    average_rho_k_cpu = cp.asnumpy(average_rho_k)
    average_rho_squared_cpu = cp.asnumpy(average_rho_squared)

    # Writing the output to a file
    #output_file_name = "epsilon_smallK.txt"
    #dk_values = args.dk
    dk_values = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    kr_max = args.kr_max
    for dk_local in dk_values:
        output_file_name = f"epsilon_smallK_{dk_local:.2f}.txt"
        with open(output_file_name, 'w') as file:
            kr_range = cp.arange(k_min, kr_max, dk_local)
            for k in kr_range:
                indices = cp.isclose(kr, k, atol=dk_local / 2)
                count = cp.sum(indices).get()  # Count how many values match this k value
                if cp.any(indices):
                    avg_rho_k = cp.mean(average_rho_k_cpu[indices.get()])
                    std_rho_k = cp.std(average_rho_k_cpu[indices.get()])
                    avg_rho_squared = cp.mean(average_rho_squared_cpu[indices.get()])
                    std_rho_squared = cp.std(average_rho_squared_cpu[indices.get()])

                    # 先计算Sk(Sk=avg_rho_squared/k/k)，再计算bin里面的平均值
                    avg_Sk1 = cp.mean(average_Sk1[indices.get()])
                    std_Sk1 = cp.std(average_Sk1[indices.get()])
                    # 直接计算chi (chi = 4*pi*beta*Sk)，再计算bin里面的平均值
                    avg_chi1 = cp.mean(average_chi1[indices.get()])
                    std_chi1 = cp.std(average_chi1[indices.get()])
                    # 先计算epsilon (epsilon = 1/(1-chi))，再计算bin里面的平均值
                    avg_eps1 = cp.mean(average_epsilon1[indices.get()])
                    std_eps1 = cp.std(average_epsilon1[indices.get()])
                    # 基于计算bin里面平均chi，再计算epsilon
                    avg_eps2 = 1 / (1 - avg_chi1)
                    std_eps2 = cp.std(average_epsilon1[indices.get()])

                    # 求了bin里面的平均值，再除以 k**2; 此处为了缩小，除以了上限。
                    avg_Sk2 = avg_rho_squared / ((k + dk_local / 2) ** 2)
                    std_Sk2 = std_rho_squared / ((k + dk_local / 2) ** 2)
                    avg_chi2 = 4 * cp.pi * 557 * avg_Sk2
                    std_chi2 = 4 * cp.pi * 557 * std_Sk2
                    ave_eps3 = 1 / (1 - avg_chi2)
                    std_eps3 = 1 / ((1 - avg_chi2) ** 2) * std_chi2
                    # 将计算结果格式化为字符串输出到文件
                    file.write(f"{k:.2f} {avg_rho_k:.6e} {avg_rho_squared:.6e} {avg_Sk1:.6e} {avg_chi1:.6e} {avg_eps2:.6e} {count}\n")

    unique_kr_values = cp.unique(kr)  # Find unique kr values
    output_file_name = f"epsilon_smallK_unique.txt"
    dk_local=1e-8
    with open(output_file_name, 'w') as file:      
        for k in unique_kr_values:
            indices = cp.isclose(kr, k, atol=dk_local)
            count = cp.sum(indices).get()  # Count how many values match this k value
            if count > 0:
                avg_rho_k = cp.mean(average_rho_k_cpu[indices.get()])
                std_rho_k = cp.std(average_rho_k_cpu[indices.get()])
                avg_rho_squared = cp.mean(average_rho_squared_cpu[indices.get()])
                std_rho_squared = cp.std(average_rho_squared_cpu[indices.get()])

                # 先计算Sk(Sk=avg_rho_squared/k/k)，再计算bin里面的平均值
                avg_Sk1 = cp.mean(average_Sk1[indices.get()])
                std_Sk1 = cp.std(average_Sk1[indices.get()])
                # 直接计算chi (chi = 4*pi*beta*Sk)，再计算bin里面的平均值
                avg_chi1 = cp.mean(average_chi1[indices.get()])
                std_chi1 = cp.std(average_chi1[indices.get()])
                # 先计算epsilon (epsilon = 1/(1-chi))，再计算bin里面的平均值
                avg_eps1 = cp.mean(average_epsilon1[indices.get()])
                std_eps1 = cp.std(average_epsilon1[indices.get()])
                # 基于计算bin里面平均chi，再计算epsilon
                avg_eps2 = 1 / (1 - avg_chi1)
                std_eps2 = cp.std(average_epsilon1[indices.get()])

                # 求了bin里面的平均值，再除以 k**2; 此处为了缩小，除以了上限。
                avg_Sk2 = avg_rho_squared / ((k + dk_local / 2) ** 2)
                std_Sk2 = std_rho_squared / ((k + dk_local / 2) ** 2)
                avg_chi2 = 4 * cp.pi * 557 * avg_Sk2
                std_chi2 = 4 * cp.pi * 557 * std_Sk2
                ave_eps3 = 1 / (1 - avg_chi2)
                std_eps3 = 1 / ((1 - avg_chi2) ** 2) * std_chi2
                # 将计算结果格式化为字符串输出到文件
                file.write(f"{k:.6f} {avg_rho_k:.6e} {avg_rho_squared:.6e} {avg_Sk1:.6e} {avg_chi1:.6e} {avg_eps2:.6e} {count}\n")


if __name__ == "__main__":
    main()

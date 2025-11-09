import argparse
import numpy as np

# Importar la biblioteca de lógica central
from symplectic_tools import bench_frame_update

# -------------------------
# Punto de Entrada Principal
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run frame-update micro-benchmarks.")
    p.add_argument("--bench-nphys", type=int, default=8192, help="Number of physical qubits n for benchmark.")
    p.add_argument("--bench-rphys", type=int, default=512, help="Number of tracked physical vectors r_phys.")
    p.add_argument("--bench-rlog", type=int, default=32, help="Number of tracked logical vectors r_log.")
    p.add_argument("--bench-loops", type=int, default=200, help="Benchmark loops.")
    p.add_argument("--uweight", type=int, default=None, help="Hamming weight of physical u (default auto ~1.5%%).")
    return p.parse_args()

def main():
    args = parse_args()

    print(">>> Benchmark for torus example (k=2, w = X1 + Z1)")
    w1 = np.array([1, 0, 1, 0], dtype=np.uint8)
    bench_frame_update(n_phys=args.bench_nphys, k=2, w_logical=w1,
                       r_phys=args.bench_rphys, r_log=args.bench_rlog,
                       loops=args.bench_loops, u_weight=args.uweight, seed=1)

    print(">>> Benchmark for punctured planar example (k=3, w = Z1 + Z2)")
    w2 = np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8)
    bench_frame_update(n_phys=args.bench_nphys, k=3, w_logical=w2,
                       r_phys=args.bench_rphys, r_log=args.bench_rlog,
                       loops=args.bench_loops, u_weight=args.uweight, seed=2)

if __name__ == "__main__":
    main()
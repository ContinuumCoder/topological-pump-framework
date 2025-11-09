# topological-pump-framework

This repository provides the implementation of the classical framework for Pauli frame updates via symplectic transvections, as used for the "topological ε-pump" described in the paper:

> **"Corners and Twists as Endpoints of an $e\leftrightarrow m$ Domain Wall: Unified Counting of Ground-State Degeneracy in the Toric Code"**

## Repository Structure

* **`symplectic_tools.py`**: A core library containing the GF(2) mathematics for creating and applying symplectic transvections (rank-one Pauli frame updates).
* **`reproduce_figures.py`**: An experiment script to generate Figures 3 and 4 from the paper, which showcase the pump's action on a torus and a punctured planar geometry.
* **`run_benchmarks.py`**: An experiment script to run the micro-benchmarks comparing the computational cost of logical vs. physical frame updates.
* **`requirements.txt`**: The necessary Python dependencies.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ContinuumCoder/topological-pump-framework.git
    cd topological-pump-framework
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Reproduce Figures

To generate Figure 3 (`fig_torus_pump_swap.png`) and Figure 4 (`fig_punctured3_pump_12.png`) from the paper:

```bash
python reproduce_figures.py
````

This will also print a detailed report of the operator mapping for each case to the console.

### 2\. Run Benchmarks

To run the performance micro-benchmarks (described in Sec. S7) [cite: 526] comparing physical vs. logical frame updates:

```bash
python run_benchmarks.py
```

You can customize the benchmark parameters using command-line arguments:

```bash
python run_benchmarks.py --bench-nphys 8192 --bench-loops 200
```

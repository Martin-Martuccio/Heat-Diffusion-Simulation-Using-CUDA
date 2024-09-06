# Heat Diffusion Simulation Using CUDA

## Overview

This project implements a heat diffusion simulation on a two-dimensional grid using CUDA to parallelize computations and improve performance. The original version of the program was sequential and executed on a single CPU. By leveraging CUDA, the goal was to accelerate computations using the GPU, with an analysis of different problem sizes and thread/block configurations to achieve optimal performance.


## Contents

- **Introduction**: Explanation of the heat diffusion problem and how CUDA is used to parallelize the solution.
- **Mathematical Concepts**: Explanation of the mathematical model for simulating heat diffusion using partial derivatives.
- **Program Analysis**: Description of the program structure, with a focus on the core function (`step kernel mod`) that was parallelized using CUDA.
- **Parallelization with CUDA**: Details of how the program was converted into a CUDA kernel, with explanations on grid/block/thread configuration.
- **Experiments**: Evaluation of execution times and speed-up for different problem sizes (1000x1000, 10,000x10,000, and 30,000x30,000) and various thread/block configurations.
- **Conclusion**: Summary of results, including optimal configurations and the overall speed-up achieved using CUDA.

## Performance Analysis
The performance of the CUDA implementation was evaluated using various configurations:

  - Problem Sizes: The heat diffusion was simulated for grid sizes of 1000x1000, 10,000x10,000, and 30,000x30,000.
  - Thread/Block Configurations: Execution time and speed-up were measured for various numbers of threads per block, particularly focusing on values that are multiples of 32, as they align with CUDA's warp size.

## Conclusion
This project successfully demonstrated the benefits of parallelization using CUDA for simulating heat diffusion in a 2D grid.

For more details on the performance analysis, graphs, and implementation specifics, please refer to the full report available in this repository. Report : [Report Link](HPC_HD.pdf)

## Authors
- [Martin Martuccio](https://github.com/Martin-Martuccio) - Project Author
- [Lorenzo Mesi](https://github.com/LorenzoMesi) - Project Author

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

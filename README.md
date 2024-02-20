# nsPipe in GPU
Our DNS code nsPipe-CUDA is the C-CUDA version of [nsPipe](https://github.com/dfeldmann/nsCouette). It solves the Navier-Stokes equations for an incompressible fluid flow in a cylindrical pipe. The governing equations for the primitive variables are discretized in a cylindrical co-ordinate system using a Fourier-Galerkin ansatz for the azimuthal and the axial direction. High-order explicit finite differences are used in the only inhomogeneous (wall-normal) direction. Periodic boundary conditions are assumed in the axial direction. nsPipe-CUDA is based on a pressure-poisson equation (PPE) formulation and a constant time-stepping. 
This is the public version of the code. If you are interested on other versions of the code please contact us and gain access to the developer repository.

## In this repository
Find the folder [Main_Code](https://github.com/Mordered/nsPipe_CUDA/tree/main/Main_Code) with the standard nsPipe_CUDA version. Find also the folder [User_Guide](https://github.com/Mordered/nsPipe_CUDA/tree/main/User_Guide), with the [User_Guide](https://github.com/Mordered/nsPipe_CUDA/blob/main/User_Guide/nsPipe_CUDA_User_Guide.pdf) of the code (still in process). 

## Dependencies and hardware/software requirements
* An NVIDIA-GPU with CUDA capabilities.
* CUDA v 2.0 or higher.
* C and NVCC compilers.
  
## Get the code
To download the code run:
```
# Download the repository
git clone https://github.com/Mordered/nsPipe-GPU.git
cd ./nsPipe_CUDA/Main_Code/
ls src
```
To build the executable, a Makefile for a standard x86_64 Linux software with C and NVCC compilers is included.

## Compile and run the code
To compile the code run 
```
make
```
wherever you have the Makefile file and the src folder. This will generate an executable. Run the executable with:
```
nohup ./Pipe &
```
## Contributors

Following is a running list of contributors in chronological order:

1. [Prof. Marc Avila](https://www.zarm.uni-bremen.de/en/research/fluid-dynamics/fluid-simulation-and-modeling.html), University of Bremen, ZARM.
2. Daniel Morón Montesdeoca, University of Bremen, ZARM.
3. Patrick Keuchel, University of Bremen, ZARM.

Specific contribution is described below:

1. Prof. Marc Avila is responsible for the numerical method/formulation and supervises the development cycle
2. Daniel Morón Montesdeoca is the main developer/maintainer and responsible for several additional features, bug-fixes, documentation and tutorials.
3. Patrick Keuchel is currently implementing a new version of the code to perform adjoint optimizations.

Other contributors are also here acknowledged
1. Dr. Alberto Vela-Martin, Department of Aerospace Engineering, Universidad Carlos III de Madrid
2. Dr. Daniel Feldmann, University of Bremen
3. [Dr. Markus Rampp](http://home.mpcdf.mpg.de/~mjr/), Max Planck Computing and Data Facility

## Documentation
Find in this repository a preliminar User Guide that is currently still being improved. For more information on the methods, check the [nsCouette user guide](https://gitlab.mpcdf.mpg.de/mjr/nscouette/blob/master/doc/nsCouetteUserGuide.pdf)

## References
If you use nsPipe-CUDA please cite:
* López, J. M., Feldmann, D., Rampp, M., Vela-Martín, A., Shi, L., & Avila, M. (2020). nsCouette–A high-performance code for direct numerical simulations of turbulent Taylor–Couette flow. SoftwareX, 11, 100395.

## Contact
If you have any questions, comments or suggestions for improvements, please fell free to contact:
* [Daniel Morón][daniel.moron@zarm.uni-bremen.de](mailto:daniel.moron@zarm.uni-bremen.de)

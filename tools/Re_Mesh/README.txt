Tool to, from a previous solution, generate a new velocity field
with the desired grid. It interpolates between the two grids 
using in the radial direction a desired polynomial degree, and
by padding or ignoring Fourier modes in the axial and azimuthal
directions

INPUTS:
- It needs three files: 'ur_o.bin', 'ut_o.bin', 'uz_o.bin' with
  the old velocity field that were generated with nsPipe CUDA
- In the head.h file, before compiling please specify:
  · NR, NT, NZ: as the old grid parameters (as in nsPipe CUDA)
  · Nrd,Ntd,Nzd: as the new grid parameters(as in nsPipe CUDA)
  · iw: half the stencil of points to use to interpolate.
    (We recommend you use half the stencil you used for 
     the radial derivatives)
- In the main.cu file, before compiling please specify in dev
  the device ID of the GPU you want to run in

OUTPUTS
- Three files with the new  and interpolated velocity field:
  'ur_s.bin', 'ut_s.bin and 'uz_s.bin'.

TO RUN:
- Once you have changed the head.h and main.cu file, please 
  compile in a cuda capable GPU
- Save the executable 'Remesh' in the same folder where you 
  have 'ur_o.bin', 'ut_o.bin' and 'uz_o.bin' files
- Run with nohup ./Remesh &


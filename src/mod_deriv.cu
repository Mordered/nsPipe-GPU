#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES for non-radial derivatives and initField 
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 18/04/2022
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*+++++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++++++*/
// KERNEL TO CALCULATE THE AZIMUTHAL DERIV....................................../
static __global__ void derivt_k(double2 *grad,double2 *p,const double *r){
   // Indexes...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x; 
   int i=blockIdx.y*blockDim.y+threadIdx.y;
   // Stopper and initialize variables
   if(k>=NT*NZ || i>=NR){return;} int it=k/NZ; it=it<NT/2 ? it:it-NT;
   // Wavenumber & global index
   double kt=(PI2/LT)*(double(it));  int h=i*NT*NZ+k; double2 u; 
   u.x=p[h].x; u.y=p[h].y;  
   // Write to auxiliary variable...........................................
   grad[h].x=-kt*u.y/r[i];  grad[h].y=kt*u.x/r[i];
}// END AZIMUTHAL DERIV........................................................./

// KERNEL TO CALCULATE THE AXIAL DERIV........................................../
static __global__ void derivz_k(double2 *grad,double2 *p){
   // Indexes...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x; 
   int i=blockIdx.y*blockDim.y+threadIdx.y;
   // Stopper and initialize variables
   if(k>=NT*NZ || i>=NR){return;} int l=k%NZ; 
   // Wavenumber & global index
   double kz=(PI2/LZ)*(double(l));  int h=i*NT*NZ+k; double2 u; 
   u.x=p[h].x; u.y=p[h].y;
   // Write to auxiliary variable...........................................
   grad[h].x=-kz*u.y; grad[h].y=kz*u.x;
}// END AXIAL DERIV............................................................./


/*++++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++++++*/
static double *r;


/*++++++++++++++++++++++++++++++ MAIN FUNCTIONS +++++++++++++++++++++++++++++++*/
// SETTER OF INTEGRATOR PART..................................................../
void setDeriv(double *r_h){
CHECK_CUDART(cudaMalloc(&r ,NR*sizeof(double)));
CHECK_CUDART(cudaMemcpy(r,r_h,NR*sizeof(double),cudaMemcpyHostToDevice)); 
}// END OF SETTER OF THE INTEGRATOR PART......................................../

// DESTROYER OF DERIVATIVES PART................................................/
void DerivDestroy(void){
 return;
}// END OF DESTROYER OF DERIVATIVES............................................./

// AZIMUTHAL DERIV............................................................../
void derivt(double2 *grad,double2 *p){
   // Dimensions
   dim3 grid,block; block.x=block_size; 
   grid.x=(NT*NZ+block.x-1)/block.x; grid.y=NR;
   // Kernel
   derivt_k<<<grid,block>>>(grad,p,r); return;
}// END OF AZIMUTHAL DERIV....................................................../

// AXIAL DERIV................................................................../
void derivz(double2 *grad,double2 *p){
   // Dimensions
   dim3 grid,block; block.x=block_size;
   grid.x=(NT*NZ+block.x-1)/block.x; grid.y=NR;
   // Kernel
   derivz_k<<<grid,block>>>(grad,p); return;
}// END OF AXIAL DERIV........................................................../



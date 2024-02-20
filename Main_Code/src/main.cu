/* MAIN FILE OF NSPIPE IN CUDA ------------------------------------------------- */
/* Purpose: nsPipe version for CUDA. It integrates the Navier-Stokes equations in 
            cylindrical coordinates using a pseudo-spectral formulation and a 
            fractional step method.
            - This version doesnot include a dynamic time step
-  Authors: Alberto Vela Martín & Daniel Morón Montesdeoca. 
-  Contact: daniel.moron@zarm.uni-bremen.de
-  Date   : 08/04/2022                                                           */ 
#include"head.h"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
int main(int argc, const char* argv[]){
 printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 printf("\n+++++++++++++++++  Welcome to nsPipe in CUDA  +++++++++++++++++");
 printf("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 // SET DEVICE AND START CLOCK
 int dev=0;  clock_t begin = clock();
 printf("\nSetting device %d\n",dev); CHECK_CUDART(cudaSetDevice(dev));
 // INITIALIZATION ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 size_p sizes; sizes.Nr=NR; sizes.Nt=NT; sizes.Nz=NZ; 
 // Initialize radial grid, derivatives and weights of integration
 double *r_h  =(double *)malloc(NR*sizeof(double));
 double *Lu_h =(double *)malloc(2*sten*NR*sizeof(double));
 double *Lp_h =(double *)malloc(2*sten*NR*sizeof(double));
 double *rdr_h=(double *)malloc(NR*sizeof(double));
 init_fd(r_h,Lu_h,Lp_h,rdr_h);
 // Setters
 setFft(sizes);  setLinear(r_h,Lu_h,Lp_h); setCublasFlux(rdr_h); setDeriv(r_h); 
 setNonlinear(); setIO(r_h);  setInt(); setBCon(); 
 // Allocate memory buffers
 vfield u, rhs;                          size_t size_p=NR*NT*NZ*sizeof(double2);
 CHECK_CUDART(cudaMalloc(&u.r,size_p));  CHECK_CUDART(cudaMalloc(&rhs.r,size_p)); 
 CHECK_CUDART(cudaMalloc(&u.t,size_p));  CHECK_CUDART(cudaMalloc(&rhs.t,size_p));
 CHECK_CUDART(cudaMalloc(&u.z,size_p));  CHECK_CUDART(cudaMalloc(&rhs.z,size_p));

 // Initialize field
 initField(u);  
 // INTEGRATE +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 integrate(u,rhs);

 // OUTRO +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 // Last velocity field
 wBufferBinary((double*)u.r,"ur.bin",sizeof(double2),NR*NT*NZ);
 wBufferBinary((double*)u.t,"ut.bin",sizeof(double2),NR*NT*NZ);
 wBufferBinary((double*)u.z,"uz.bin",sizeof(double2),NR*NT*NZ);
 // Destroyers of the Universe
 fftDestroy(); LinearDestroy(); FluxDestroy(); DerivDestroy(); BConDestroy();
 NonDestroy(); IODestroy();     IntDestroy();  
 // Free GPU memory
 cudaFree(u.r);      cudaFree(u.t);      cudaFree(u.z);
 cudaFree(rhs.r);    cudaFree(rhs.t);    cudaFree(rhs.z);
 // Free CPU memory
 free(r_h); free(Lu_h); free(Lp_h); free(rdr_h);
 // Final time and print
 clock_t end=clock();        double tot_t=(double)(end-begin)/CLOCKS_PER_SEC;
 printf("\n Total Time = %.3fs\n",tot_t);          printf("\n");    return 0;
}// END THE CODE!


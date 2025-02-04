/* MAIN FILE OF RE_MESHING IN NSPIPE CUDA ------------------------------- */
/* Purpose: Tool to increase/decrease the number of points in the domain
-  Authors: Daniel Morón Montesdeoca. 
-  Contact: daniel.moron@zarm.uni-bremen.de
-  Date   : 08/04/2022                                                   */ 
#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
int main(int argc, const char* argv[]){
 printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 printf("\n++++++++  Welcome to re-mesh nsPipe in CUDA  +++++++++++++++++");
 printf("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
 // SET DEVICE AND START CLOCK
 int dev=1;  clock_t begin = clock();
 printf("\nSetting device %d\n",dev); CHECK_CUDART(cudaSetDevice(dev));
 // INITIALIZATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 // Allocate memory buffers
 vfield u;              size_t size_p=Nrd*Ntd*Nzd*sizeof(double2);
 CHECK_CUDART(cudaMalloc(&u.r,size_p));   
 CHECK_CUDART(cudaMalloc(&u.t,size_p));  
 CHECK_CUDART(cudaMalloc(&u.z,size_p));  
 // Initialize the value of ur, ut, uz
 double2 *vec=(double2 *)malloc(size_p);  int N2=Nrd*Ntd*Nzd;
 for(int i=0;i<Nrd*Ntd*Nzd;i++){vec[i].x=0.0; vec[i].y=0.0;}
CHECK_CUDART(cudaMemcpy(u.r,vec,N2*sizeof(double2),cudaMemcpyHostToDevice));CHECK_CUDART(cudaMemcpy(u.t,vec,N2*sizeof(double2),cudaMemcpyHostToDevice));
CHECK_CUDART(cudaMemcpy(u.z,vec,N2*sizeof(double2),cudaMemcpyHostToDevice));
 free(vec); 

 // RE-MESH TOOL-----------------------------------------------------
 re_mesh(u); 
 
 // Last velocity field
 wBufferBinary((double*)u.r,"ur_s.bin",sizeof(double2),Nrd*Ntd*Nzd);
 wBufferBinary((double*)u.t,"ut_s.bin",sizeof(double2),Nrd*Ntd*Nzd);
 wBufferBinary((double*)u.z,"uz_s.bin",sizeof(double2),Nrd*Ntd*Nzd);
 // Free GPU memory
 cudaFree(u.r);      cudaFree(u.t);      cudaFree(u.z);
 // Final time and print
 clock_t end=clock();      double tot_t=(double)(end-begin)/CLOCKS_PER_SEC;
 printf("\n Total Time = %.3fs\n",tot_t);        printf("\n");    return 0;
}// END THE CODE!


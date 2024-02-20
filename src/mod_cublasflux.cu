#include"head.h"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to calculate the flux correction & CUBLAS
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 11/04/2022
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*++++++++++++++++++++++++++++++++++ KERNELS ++++++++++++++++++++++++++++++++++*/
// KERNEL TO ADJUST FLUX......................................................../
static __global__ void adj_flux_k(double2 *uz,double *uc,double corr){
   // Index & Stopper & Index
   int i=blockIdx.x*blockDim.x + threadIdx.x; if(i>=NR){return;} int h=i*NT*NZ;
   // Add to the velocity
   uz[h].x+=corr*uc[i];
}// END KERNEL TO ADJUST FLUX.................................................../


// KERNEL TO WRITE 1ST MODE TO VECTOR.........................................../
static __global__ void copy1stM_k(double *uco,double2 *uaux){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    // Stopper & Computation
    if(h>=NR){return;}  uco[h]=uaux[h*NT*NZ].x;
}// END KERNEL TO WRITE 1ST MODE TO VECTOR....................................../


/*++++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++++++*/
static double *ua,*uc,*rdr,*Qa_h,*Qc_h;  static cublasHandle_t cublasHandle;
static void copy1stM(double *uc,double2 *uaux);

/*+++++++++++++++++++++++++++++++ MAIN FUNCTIONS ++++++++++++++++++++++++++++++*/
// CUBLAS CHECK----------------------------------------------------------------*/
void cublasCheck(cublasStatus_t error, const char* function ){
  if(error !=  CUBLAS_STATUS_SUCCESS){
    printf("\n error  %s : %d \n", function, error);  exit(1);
  }
  return;
}// END CUBLAS CHECK-----------------------------------------------------------*/


// PERFORM DOT PRODUCT USING CUBLAS---------------------------------------------/
void dotCub(double *Q,double* u){
    cublasCheck(cublasDdot(cublasHandle,NR,rdr,1,u,1,Q),"Tr");   return;
}// END OF DOT PRODUCT USING CUBLAS............................................./


// WRAPPER TO COPY JUST 1 MODE TO THE DESIRED VECTOR............................/
void copy1stM(double *uc,double2 *uaux){
    // Dimension of the block and grid
    dim3 grid,block; block.x=block_size; grid.x=(NR+block.x-1)/block.x;
    copy1stM_k<<<grid,block>>>(uc,uaux);         return;
}// END WRAPPER TO COPY 1st MODE................................................/


// SETTER CUBLAS AND FLUX FUNCTIONS---------------------------------------------/
void setCublasFlux(double *rdr_h){
    // INITIALIZE Setting the size of buffers
    size_t size=NR*NT*NZ*sizeof(double2); double2 *aux, *uaux;
    // Allocate uc,rdr and auxiliary functions
    CHECK_CUDART(cudaMalloc(&aux,size));
    CHECK_CUDART(cudaMalloc(&uaux,size)); 
    CHECK_CUDART(cudaMalloc(&uc ,NR*sizeof(double)));
    CHECK_CUDART(cudaMalloc(&ua ,NR*sizeof(double)));
    CHECK_CUDART(cudaMalloc(&rdr,NR*sizeof(double)));
    // CudaMemcpy
    CHECK_CUDART(cudaMemcpy(rdr,rdr_h,NR*sizeof(double),cudaMemcpyHostToDevice));
    // Cublacreate
    cublasCheck(cublasCreate(&cublasHandle),"Cre");
    // COMPUTE THE U_A VECTOR..........................
    double2 uni,bcon; uni.x=1.0; uni.y=0.0; bcon.x=0.0; bcon.y=0.0;
    // RHS
    uniformVec(aux,uni); write_bcon(aux,bcon);
    // Solve the system
    LUsolve_u(uaux,aux,2);
    // Copy to uc only 
    copy1stM(uc,uaux); cudaFree(aux); cudaFree(uaux);
    // Dot product
    Qc_h=(double *)malloc(NR*sizeof(double));  dotCub(Qc_h,uc); 
    Qa_h=(double *)malloc(NR*sizeof(double));  return; 
}//END OF SETTER AND FLUX....................................................../


// DESTROYER OF CUBLAS AND FLUX................................................/
void FluxDestroy(void){
   cudaFree(uc); cudaFree(ua);  cudaFree(rdr);  free(Qc_h);  free(Qa_h); return;
}// END OF DESTROYER OF CUBLAS AND FLUX......................................../


// ADJUST THE FLUX............................................................./
void adj_flux(double2 *uz){
   // Define variables
   double corr,Qdes=0.5;
   // Copy 1st Mode
   copy1stM(ua,uz); 
   // Compute the dot product
   dotCub(Qa_h,ua); 
   // Compute the correction  
   corr=(Qdes-Qa_h[0])/Qc_h[0];
   // Add the correction to u.z
   dim3 block,grid; block.x=block_size; grid.x=(NR+block.x-1)/block.x;
   adj_flux_k<<<grid,block>>>(uz,uc,corr);  
   return;
}// END OF ADJUST FLUX........................................................../

// IDENTIFY THE MAXIMUM........................................................./
void iden_max(double *err,double *d1,double *d2){
   // Initialize variables
   double num,den; int maxIndex;
   // Identify the location of the maximum numerator
cublasCheck(cublasIdamax(cublasHandle,NR*NT*NZ,(double *)d1,1,&maxIndex),"Tr");
CHECK_CUDART(cudaMemcpy(&num,d1+maxIndex-1,sizeof(double),cudaMemcpyDeviceToHost));
   // Identify the location of the maximum denominator
cublasCheck(cublasIdamax(cublasHandle,NR*NT*NZ,(double *)d2,1,&maxIndex),"Tr");
CHECK_CUDART(cudaMemcpy(&den,d2+maxIndex-1,sizeof(double),cudaMemcpyDeviceToHost));
    err[0]=num/den; return;
}// END OF IDENTIFY THE MAXIMUM................................................../

// INTEGRATE Q.................................................................../
void intq(double *qin,double *q,double *qaux){
  // Initialize variables
  double *q1,al=1.0,be=0.0; CHECK_CUDART(cudaMalloc(&q1,NR*2*NZP*sizeof(double)));
  // Perform 1st Integration in th as a Matrix*Vector multiplication
  cublasCheck(cublasDgemv(cublasHandle,CUBLAS_OP_N,NR*2*NZP,NTP,&al,q,NR*2*NZP,
                          qaux,1,&be,q1,1),"Tr");
  // Perform 2nd Integration in r as a Matrix*Vector multiplication
  cublasCheck(cublasDgemv(cublasHandle,CUBLAS_OP_N,2*NZP,NR,&al,q1,2*NZP,
                          rdr,1,&be,qin,1),"Tr");
  // Deallocate
  cudaFree(q1);  return;
}// END OF INTEGRATE Q............................................................/

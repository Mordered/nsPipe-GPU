#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES from Alberto's code that complement others
// AUTHOR: ----------- Alberto Vela Martín
// DATE:   ----------- 27/03/2022
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*++++++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++++*/
// KERNEL TO NORMALIZE........................................................
static __global__ void normalize_k(double2 *u,double norm,size_t elements){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    // Stopper
    if(h>=elements){return;}
    // Normalize
    double2 uh=u[h]; uh.x=uh.x/norm; uh.y=uh.y/norm; u[h]=uh;
}// END KERNEL TO NORMALIZE..................................................

// KERNEL TO WRITE BCON TO RHS...............................................
static __global__ void write_bcon_k(double2 *rhs, double2 bcon){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x; 
    // Stopper
    if(h>=NT*NZ){return;} int i=(NR-1)*NT*NZ+h;
    // Write the Bcon
    rhs[i].x=bcon.x; rhs[i].y=bcon.y;
}// END KERNEL TO WRITE BCON..................................................

// KERNEL TO DECOUPLE FORWARD.................................................
static __global__ void decoupleForward_k(double2 *ur,double2 *ut){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    // Stopper
    if(h>=NR*NT*NZ){return;} double2 ur_l,ut_l,up_l,um_l;
    // Copy to kernel
    ur_l=ur[h];  ut_l=ut[h]; 
    // Write to local variables +/- velocity
    up_l.x= ur_l.x - ut_l.y;  up_l.y= ur_l.y + ut_l.x;
    um_l.x= ur_l.x + ut_l.y;  um_l.y= ur_l.y - ut_l.x;
    // Write back to velocity field
    ur[h]=up_l;  ut[h]=um_l;
}// END KERNEL TO DECOUPLE FORWARD............................................

// KERNEL TO DECOUPLE BACKWARD................................................
static __global__ void decoupleBackward_k(double2 *up,double2 *um){
    // Index
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    // Stopper
    if(h>=NR*NT*NZ){return;} double2 ur_l,ut_l,up_l,um_l;
    // Copy to kernel
    up_l=up[h];  um_l=um[h];
    // Write to local variables r/t velocity
    ur_l.x=  0.5*(up_l.x + um_l.x);   ur_l.y=  0.5*(up_l.y + um_l.y);
    ut_l.x=  0.5*(up_l.y - um_l.y);   ut_l.y= -0.5*(up_l.x - um_l.x);
    // Write back to velocity field
    up[h]=ur_l;  um[h]=ut_l;
}// END KERNEL TO DECOUPLE FORWARD............................................


/*++++++++++++++++++++++++++++++ MAIN FUNCTIONS +++++++++++++++++++++++++++++*/
// WRAPPER TO NORMALIZE BY A GIVEN NORM......................................
void normalize(double2 *u,double norm,size_t elements){
    // Dimension of the block and grid
    dim3 grid,block; block.x=block_size; grid.x=(elements+block.x-1)/block.x;
    // Call the kernel
    normalize_k<<<grid,block>>>(u, norm, elements);  return;
}// END WRAPPER TO NORMALIZE A GIVEN VECTOR...................................

// WRAPPER TO INTRODUCE BCON TO A GIVEN  RHS VECTOR...........................
void write_bcon(double2 *rhs,double2 bcon){
    // Dimension of the block and grid
    dim3 grid,block; block.x=block_size; grid.x=(NT*NZ+block.x-1)/block.x;
    write_bcon_k<<<grid,block>>>(rhs, bcon);     return;
}// END WRAPPER TO BCON TO A GIVEN RHS VECTOR.................................

// UNIFORM VECTOR DEFINITION..................................................
void uniformVec(double2 *u,double2 uni){
    // Initialize vector
    double2* vec=(double2*)malloc(NR*NT*NZ*sizeof(double2));
    // Loop
    for(int i=0;i<NR*NT*NZ;i++){vec[i].x=uni.x; vec[i].y=uni.y;}
    // Copy back to CUDA
CHECK_CUDART(cudaMemcpy(u,vec,NR*NT*NZ*sizeof(double2),cudaMemcpyHostToDevice)); 
    free(vec);  return;
}// END UNIFORM VECTOR DEFINITION..............................................

// WRAPPER TO DECOUPLE FORWARD from r/t to +/-.................................
void decoupleForward(vfield u){
    // Dimension of the block and grid
    dim3 grid,block; block.x=block_size; grid.x=(NR*NT*NZ+block.x-1)/block.x;
    decoupleForward_k<<<grid,block>>>(u.r,u.t); return;
}// END WRAPPER DECOUPLE FORWARD...............................................

// WRAPPER TO DECOUPLE BACKWARD from +/- to r/t................................
void decoupleBackward(vfield u){
    // Dimension of the block and grid
    dim3 grid,block; block.x=block_size; grid.x=(NR*NT*NZ+block.x-1)/block.x;
    decoupleBackward_k<<<grid,block>>>(u.r,u.t); return;
}// END WRAPPER COUPLE FORWARD.................................................

// WRAPPER TO COPY ONE VECTOR (u1->u2).........................................
void copyBuffer(double2* u2, double2* u1){
    size_t size=NR*NT*NZ*sizeof(double2);
    // Copy
    CHECK_CUDART(cudaMemcpy(u2,u1,size,cudaMemcpyDeviceToDevice)); return;
}// END WRAPPER TO COPY ONE VECTOR.............................................

// WRAPPER TO COPY THE VELOCITY FIELD (u1->u2).................................
void copyVfield(vfield u2, vfield u1){
    size_t size=NR*NT*NZ*sizeof(double2);
    // Copy
    CHECK_CUDART(cudaMemcpy(u2.r,u1.r,size,cudaMemcpyDeviceToDevice));
    CHECK_CUDART(cudaMemcpy(u2.t,u1.t,size,cudaMemcpyDeviceToDevice));
    CHECK_CUDART(cudaMemcpy(u2.z,u1.z,size,cudaMemcpyDeviceToDevice));  return;
}// END WRAPPER TO COPY THE VELOCITY FIELD.....................................

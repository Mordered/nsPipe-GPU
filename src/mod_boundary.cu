#include"head.h"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to calculate auxiliary variables for the BC.
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 11/04/2022
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/


/*++++++++++++++++++++++++++++++++++ KERNELS ++++++++++++++++++++++++++++++++++*/
// KERNEL TO ADD THE HOMOGENEOUS VELOCITIES...................................../
static __global__ void addBCon_k(double2 *ur,double2 *ut,double2 *uz,
                   double2 *coef,double2 *u_hr,double2 *u_ht,double2 *u_hz,
                                 double2 *u_hPr,double2 *u_hPt,double2 *u_hPz){
  // Index...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; 
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper and index 
  if(k>=NT*NZ || i>=NR){return;} int h=i*NT*NZ+k; double2 aux;
  // Read coeff
  double2 a0=coef[0+4*k],  a1=coef[1+4*k],  a2=coef[2+4*k],  a3=coef[3+4*k];
  // Read Hom velocities
  double2 uhr=u_hr[h],     uht=u_ht[h],     uhz=u_hz[h];
  // Read Hom velocities of pressure
  double2 uhPr=u_hPr[h],   uhPt=u_hPt[h],   uhPz=u_hPz[h];
  // Uplus................................................................
  aux.x   = a0.x*uhr.x - a0.y*uhr.y + a3.x*uhPr.x - a3.y*uhPr.y;
  aux.y   = a0.x*uhr.y + a0.y*uhr.x + a3.x*uhPr.y + a3.y*uhPr.x;
  ur[h].x+= aux.x;          ur[h].y+= aux.y; 
  // Uminus................................................................
  aux.x   = a1.x*uht.x - a1.y*uht.y + a3.x*uhPt.x - a3.y*uhPt.y;
  aux.y   = a1.x*uht.y + a1.y*uht.x + a3.x*uhPt.y + a3.y*uhPt.x;
  ut[h].x+= aux.x;          ut[h].y+= aux.y;
  // Uz....................................................................
  aux.x   = a2.x*uhz.x - a2.y*uhz.y + a3.x*uhPz.x - a3.y*uhPz.y;
  aux.y   = a2.x*uhz.y + a2.y*uhz.x + a3.x*uhPz.y + a3.y*uhPz.x;
  uz[h].x+= aux.x;          uz[h].y+= aux.y;  
}// END KERNEL TO ADD THE HOMOGENEOUS VELOCITIES................................/



// KERNEL TO COMPUTE THE COEF VECTOR............................................/
static __global__ void compcoef_k(double2 *coef,double2 *div1,double2 *invM){
  // Index...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  // Stopper
  if(k>=NT*NZ){return;}
  // Read variables
  double2 dv, a0, a1, a2, a3; 
  dv.x = -div1[k].x; dv.y = -div1[k].y;
  double2 IM0=invM[3+16*k],IM1=invM[7+16*k],IM2=invM[11+16*k],IM3=invM[15+16*k];
  // Compute the coefficients............................................
  // 0 coef
  a0.x = dv.x*IM0.x - dv.y*IM0.y;   a0.y = dv.x*IM0.y + dv.y*IM0.x;
  // 1 coef
  a1.x = dv.x*IM1.x - dv.y*IM1.y;   a1.y = dv.x*IM1.y + dv.y*IM1.x;
  // 2 coef
  a2.x = dv.x*IM2.x - dv.y*IM2.y;   a2.y = dv.x*IM2.y + dv.y*IM2.x;
  // 3 coef
  a3.x = dv.x*IM3.x - dv.y*IM3.y;   a3.y = dv.x*IM3.y + dv.y*IM3.x;
  // Save to coef
  coef[0+4*k]=a0;  coef[1+4*k]=a1;  coef[2+4*k]=a2;  coef[3+4*k]=a3;
}// END KERNEL TO COMPUTE THE COEF VECTOR....................................../


// KERNEL OF THE INVERSE OF THE INFLUENCE MATRIX.............................../
static __global__ void calc_invM_k(double2 *invM,double2 *div1,double2 *div2,
              double2 *div3,double2 *div4,double2 *up,double2 *um,double2 *uz){
  // Index...............................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x;  
  // Stopper & Index in global vector
  if(k>=NT*NZ){return;} int h=(NR-1)*NT*NZ+k;  
  // Write values........................................................
  double2 a14,a24,a34,a44,a41,a42,a43,Det,aux;
  double2 IA0, IA1, IA2, IA3, IA4, IA5, IA6, IA7, IA8, IA9;
  double2 IA10,IA11,IA12,IA13,IA14,IA15;
  // Last column
  a14=up[h];   a24=um[h];   a34=uz[h];   a44=div4[k];
  // Last row
  a41=div1[k]; a42=div2[k]; a43=div3[k];
  //Determinant.........................................................
  aux.x=  a44.x - a24.x*a42.x + a24.y*a42.y - a14.x*a41.x + a14.y*a41.y;
  aux.y=  a44.y - a24.x*a42.y - a24.y*a42.x - a14.x*a41.y - a14.y*a41.x;
  Det.x=- aux.y - a34.x*a43.x + a34.y*a43.y;
  Det.y=  aux.x - a34.x*a43.y - a34.y*a43.x;
  //Row 0 of inverse matrix.............................................
    // Column 0
    aux.x   =  a44.x - a24.x*a42.x + a24.y*a42.y;
    aux.y   =  a44.y - a24.x*a42.y - a24.y*a42.x;
    IA0.x =- aux.y - a34.x*a43.x + a34.y*a43.y;
    IA0.y =  aux.x - a34.x*a43.y - a34.y*a43.x;
    // Column 1
    IA1.y =  a14.x*a42.x - a14.y*a42.y;
    IA1.x =- a14.x*a42.y - a14.y*a42.x;
    // Column 2
    IA2.x =  a14.x*a43.x - a14.y*a43.y;
    IA2.y =  a14.x*a43.y + a14.y*a43.x;
    // Column 3
    IA3.x =  a14.y;
    IA3.y =- a14.x; 
  //Row 1 of inverse matrix............................................
    // Column 0
    IA4.y =  a24.x*a41.x - a24.y*a41.y;
    IA4.x =- a24.x*a41.y - a24.y*a41.x;
    // Column 1
    aux.x   =  a44.x - a14.x*a41.x + a14.y*a41.y;
    aux.y   =  a44.y - a14.x*a41.y - a14.y*a41.x;
    IA5.x =- aux.y - a34.x*a43.x + a34.y*a43.y;
    IA5.y =  aux.x - a34.x*a43.y - a34.y*a43.x;
    // Column 2
    IA6.x =  a24.x*a43.x - a24.y*a43.y;
    IA6.y =  a24.x*a43.y + a24.y*a43.x;
    // Column 3
    IA7.x =  a24.y;
    IA7.y =- a24.x;
  //Row 2 of inverse matrix............................................
    // Column 0
    IA8.x =  a34.x*a41.x - a34.y*a41.y;
    IA8.y =  a34.x*a41.y + a34.y*a41.x;
    // Column 1
    IA9.x =  a34.x*a42.x - a34.y*a42.y;
    IA9.y =  a34.x*a42.y + a34.y*a42.x;
    // Column 2
    IA10.x=  a44.x - a14.x*a41.x + a14.y*a41.y - a24.x*a42.x + a24.y*a42.y;
    IA10.y=  a44.y - a14.x*a41.y - a14.y*a41.x - a24.x*a42.y - a24.y*a42.x;
    // Column 3
    IA11.x=- a34.x;
    IA11.y=- a34.y; 
  //Row 3 of inverse matrix.............................................
    // Column 0
    IA12.x=  a41.y;
    IA12.y=- a41.x;
    // Column 1
    IA13.x=  a42.y;
    IA13.y=- a42.x;
    // Column 2
    IA14.x=- a43.x;
    IA14.y=- a43.y;
    // Column 3
    IA15.x= 0.0;
    IA15.y= 1.0;
  //Divide by the determinant............................................
  double D1=Det.x*Det.x + Det.y*Det.y;  int ind;
  Det.x=Det.x/D1; Det.y=-Det.y/D1;
  // Row 0 of inverse matrix.............................................
  // Column 0 (0)
  aux.x=IA0.x*Det.x - IA0.y*Det.y;  ind=0;
  IA0.y=IA0.x*Det.y + IA0.y*Det.x;  IA0.x=aux.x;  invM[ind+k*16]=IA0; 
  // Column 1 (1)
  aux.x=IA1.x*Det.x - IA1.y*Det.y;  ind=1;
  IA1.y=IA1.x*Det.y + IA1.y*Det.x;  IA1.x=aux.x;  invM[ind+k*16]=IA1;
  // Column 2 (2)
  aux.x=IA2.x*Det.x - IA2.y*Det.y;  ind=2;
  IA2.y=IA2.x*Det.y + IA2.y*Det.x;  IA2.x=aux.x;  invM[ind+k*16]=IA2;
  // Column 3 (3)
  aux.x=IA3.x*Det.x - IA3.y*Det.y;  ind=3;
  IA3.y=IA3.x*Det.y + IA3.y*Det.x;  IA3.x=aux.x;  invM[ind+k*16]=IA3;
  // Row 1 of inverse matrix.............................................
  // Column 0 (4)
  aux.x=IA4.x*Det.x - IA4.y*Det.y;  ind=4;
  IA4.y=IA4.x*Det.y + IA4.y*Det.x;  IA4.x=aux.x;  invM[ind+k*16]=IA4;
  // Column 1 (5)
  aux.x=IA5.x*Det.x - IA5.y*Det.y;  ind=5;
  IA5.y=IA5.x*Det.y + IA5.y*Det.x;  IA5.x=aux.x;  invM[ind+k*16]=IA5;
  // Column 2 (6)
  aux.x=IA6.x*Det.x - IA6.y*Det.y;  ind=6;
  IA6.y=IA6.x*Det.y + IA6.y*Det.x;  IA6.x=aux.x;  invM[ind+k*16]=IA6;
  // Column 3 (7)
  aux.x=IA7.x*Det.x - IA7.y*Det.y;  ind=7;
  IA7.y=IA7.x*Det.y + IA7.y*Det.x;  IA7.x=aux.x;  invM[ind+k*16]=IA7;
  // Row 2 of inverse matrix.............................................
  // Column 0 (8)
  aux.x=IA8.x*Det.x - IA8.y*Det.y;  ind=8;
  IA8.y=IA8.x*Det.y + IA8.y*Det.x;  IA8.x=aux.x;  invM[ind+k*16]=IA8;
  // Column 1 (9)
  aux.x=IA9.x*Det.x - IA9.y*Det.y;  ind=9;
  IA9.y=IA9.x*Det.y + IA9.y*Det.x;  IA9.x=aux.x;  invM[ind+k*16]=IA9;
  // Column 2 (10)
  aux.x =IA10.x*Det.x - IA10.y*Det.y;  ind=10;
  IA10.y=IA10.x*Det.y + IA10.y*Det.x;  IA10.x=aux.x;  invM[ind+k*16]=IA10;
  // Column 3 (11)
  aux.x =IA11.x*Det.x - IA11.y*Det.y;  ind=11;
  IA11.y=IA11.x*Det.y + IA11.y*Det.x;  IA11.x=aux.x;  invM[ind+k*16]=IA11;
  // Row 3 of inverse matrix.............................................
  // Column 0 (12)
  aux.x =IA12.x*Det.x - IA12.y*Det.y;  ind=12;
  IA12.y=IA12.x*Det.y + IA12.y*Det.x;  IA12.x=aux.x;  invM[ind+k*16]=IA12;
  // Column 1 (13)
  aux.x =IA13.x*Det.x - IA13.y*Det.y;  ind=13;
  IA13.y=IA13.x*Det.y + IA13.y*Det.x;  IA13.x=aux.x;  invM[ind+k*16]=IA13;
  // Column 2 (14)
  aux.x =IA14.x*Det.x - IA14.y*Det.y;  ind=14;
  IA14.y=IA14.x*Det.y + IA14.y*Det.x;  IA14.x=aux.x;  invM[ind+k*16]=IA14;
  // Column 3 (15)
  aux.x =IA15.x*Det.x - IA15.y*Det.y;  ind=15;
  IA15.y=IA15.x*Det.y + IA15.y*Det.x;  IA15.x=aux.x;  invM[ind+k*16]=IA15;
}// END KERNEL TO COMPUTE THE INVERSE OF THE INFLUENCE MATRIX................../


/*+++++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++++*/
static vfield u_h,u_hP; static double2 *invM, *div1, *coef;


/*++++++++++++++++++++++++++++++++ MAIN FUNCTIONS ++++++++++++++++++++++++++++*/
// SETTER....................................................................../
void setBCon(void){
  size_t size=NR*NT*NZ*sizeof(double2);
  // Allocate auxiliary variables needed to correct the velocity...............
  CHECK_CUDART(cudaMalloc((void**)&u_h.r,size));
  CHECK_CUDART(cudaMalloc((void**)&u_h.t,size));
  CHECK_CUDART(cudaMalloc((void**)&u_h.z,size));
  // Homogeneous solution 4
  CHECK_CUDART(cudaMalloc((void**)&u_hP.r,size));
  CHECK_CUDART(cudaMalloc((void**)&u_hP.t,size));
  CHECK_CUDART(cudaMalloc((void**)&u_hP.z,size));
  CHECK_CUDART(cudaMalloc((void**)&invM,4*4*NT*NZ*sizeof(double2)));
  CHECK_CUDART(cudaMalloc((void**)&coef,4*NT*NZ*sizeof(double2)));
  // Define variables needed for the setter
  double2 *rhs,*aux,*div2,*div3,*div4;
  // Allocate the variables needed for the setter
  CHECK_CUDART(cudaMalloc(&rhs,size));
  CHECK_CUDART(cudaMalloc(&aux,size));
  CHECK_CUDART(cudaMalloc((void**)&div1,NT*NZ*sizeof(double2)));
  CHECK_CUDART(cudaMalloc(&div2,NT*NZ*sizeof(double2)));
  CHECK_CUDART(cudaMalloc(&div3,NT*NZ*sizeof(double2)));
  CHECK_CUDART(cudaMalloc(&div4,NT*NZ*sizeof(double2)));
  double2 uni,bcon; uni.x=0.0; uni.y=0.0; bcon.x=1.0; bcon.y=0.0;
  // U1: SOLVE THE PLUS HOMOGENEOUS VELOCITY...................................
  uniformVec(rhs,uni);  write_bcon(rhs,bcon); LUsolve_u(u_h.r,rhs,0); 
  // U2: SOLVE THE MINUS HOMOGENEOUS VELOCITY
  LUsolve_u(u_h.t,rhs,1); bcon.x=0.0; bcon.y=1.0; uniformVec(rhs,uni); 
  // U3: SOLVE THE AXIAL HOMOGENEOUS VELOCITY
  write_bcon(rhs,bcon); LUsolve_u(u_h.z,rhs,2); uniformVec(rhs,uni);
  // U4: SOLVE THE PRESSURE HOMOGENEOUS
  bcon.x=-1.0; bcon.y=0.0; write_bcon(rhs,bcon); LUsolve_p(aux,rhs); 
  // DIV1: FIRST DIVERGENCE (only up)..........................................
  copyVfield(u_hP,u_h); uniformVec(u_hP.t,uni); uniformVec(u_hP.z,uni); 
  decoupleBackward(u_hP); 
  divWall(div1,u_hP);  
  // DIV2: SECOND DIVERGENCE (only um)
  copyVfield(u_hP,u_h); uniformVec(u_hP.r,uni); uniformVec(u_hP.z,uni);
  decoupleBackward(u_hP); 
  divWall(div2,u_hP);   
  // DIV3: THIRD DIVERGENCE (only uz)
  copyVfield(u_hP,u_h); uniformVec(u_hP.r,uni); uniformVec(u_hP.t,uni);
  divWall(div3,u_hP);   uniformVec(u_hP.z,uni);
  // DIV4: FOURTH DIVERGENCE and u_hP
  derivr(u_hP.r,aux,0); derivt(u_hP.t,aux);  derivz(u_hP.z,aux); 
  divWall(div4,u_hP);   decoupleForward(u_hP);  
  // Call kernel to calculate the inverse of the influence matrix
  dim3 grid,block; block.x=block_size; grid.x=(NT*NZ+block.x-1)/block.x; 
  calc_invM_k<<<grid,block>>>(invM,div1,div2,div3,div4,u_hP.r,u_hP.t,u_hP.z);
 // Dellocate the variables of the setter.....................................
  cudaFree(div2);  cudaFree(div3);  cudaFree(div4);
  cudaFree(rhs);   cudaFree(aux);   return;
}// END SETTER................................................................./



// DESTROYER.................................................................../
void BConDestroy(void){
  cudaFree(u_h.r);  cudaFree(u_h.t);  cudaFree(u_h.z);
  cudaFree(u_hP.r); cudaFree(u_hP.t); cudaFree(u_hP.z);
  cudaFree(invM);   cudaFree(div1);   cudaFree(coef);    return;
}// END DESTROYER............................................................../



// CORRECT BOUNDARY CONDITION................................................../
void corBCon(vfield u){
   // Calculate the divergence at the wall (for u in r/th)
   decoupleBackward(u); divWall(div1,u); decoupleForward(u);
   // Compute the coef for each wavenumber
   dim3 block,grid; block.x=block_size;  grid.x=(NT*NZ+block.x-1)/block.x;
   compcoef_k<<<grid,block>>>(coef,div1,invM);    grid.y=NR;
   // Sum up the homogeneous solutions to the velocity field
   addBCon_k<<<grid,block>>>(u.r,u.t,u.z,coef,u_h.r, u_h.t, u_h.z,
                                              u_hP.r,u_hP.t,u_hP.z); return;
}// END CORRECT BOUNDARY CONDITION............................................./




  /*double2 *d1_h=(double2 *)malloc(NT*NZ*sizeof(double2));
  double2 *d2_h=(double2 *)malloc(NT*NZ*sizeof(double2));
  double2 *d3_h=(double2 *)malloc(NT*NZ*sizeof(double2));
  double2 *d4_h=(double2 *)malloc(NT*NZ*sizeof(double2));
  double2 *up  =(double2 *)malloc(NR*NT*NZ*sizeof(double2));
  double2 *um  =(double2 *)malloc(NR*NT*NZ*sizeof(double2));
  double2 *uz  =(double2 *)malloc(NR*NT*NZ*sizeof(double2));
  double2 *iM_h=(double2 *)malloc(16*NT*NZ*sizeof(double2));
  cudaMemcpy(d1_h,div1,NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  cudaMemcpy(d2_h,div2,NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  cudaMemcpy(d3_h,div3,NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  cudaMemcpy(d4_h,div4,NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  cudaMemcpy(up,u_hP.r,NR*NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  cudaMemcpy(um,u_hP.t,NR*NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  cudaMemcpy(uz,u_hP.z,NR*NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  cudaMemcpy(iM_h,invM,16*NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  for(int i=0;i<NT*NZ;i++){printf("\n d1[%d]=%e+i%e",i+1,d1_h[i].x,d1_h[i].y);}
  for(int i=0;i<NT*NZ;i++){printf("\n d2[%d]=%e+i%e",i+1,d2_h[i].x,d2_h[i].y);}
  for(int i=0;i<NT*NZ;i++){printf("\n d3[%d]=%e+i%e",i+1,d3_h[i].x,d3_h[i].y);}
  for(int i=0;i<NT*NZ;i++){printf("\n d4[%d]=%e+i%e",i+1,d4_h[i].x,d4_h[i].y);}
  int ii=(NR-1)*NT*NZ;
  for(int i=0;i<NT*NZ;i++){printf("\n up[%d]=%e+i%e",i+1,up[ii+i].x,up[ii+i].y);}
  for(int i=0;i<NT*NZ;i++){printf("\n um[%d]=%e+i%e",i+1,um[ii+i].x,um[ii+i].y);}
  for(int i=0;i<NT*NZ;i++){printf("\n uz[%d]=%e+i%e",i+1,uz[ii+i].x,uz[ii+i].y);}
  for(int j=0;j<NT*NZ;j++){for(int i=0;i<16;i++){   
    printf("\n invM[%d,%d]=%e+i%e",i+1,j+1,iM_h[16*j+i].x,iM_h[16*j+i].y);}}
  double2 *p_h  =(double2 *)malloc(NR*NT*NZ*sizeof(double2));
  cudaMemcpy(p_h,aux,NR*NT*NZ*sizeof(double2),cudaMemcpyDeviceToHost);
  for(int i=0;i<NR;i++){
   for(int j=0;j<NT*NZ;j++){
      printf("\n p(%d,%d)=%e + 1i*(%e)",i+1,j+1,p_h[i*NT*NZ+j].x,p_h[i*NT*NZ+j].y);   }
  }
  */

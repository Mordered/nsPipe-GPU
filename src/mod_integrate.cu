#include"head.h"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to perform the integration and convergence
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 28/04/2022
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*++++++++++++++++++++++++++++++++++ KERNELS ++++++++++++++++++++++++++++++++++*/
// KERNEL FOR MEASURE CORRECTION .............................................../
static __global__ void mcor_k(double *d1,  double *d2,double2 *u1r,double2 *u1t,
                            double2 *u1z,double2 *u2r,double2 *u2t,double2 *u2z){
  // Index........................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; int i=blockIdx.y;
  // Stopper
  if(k>=NT*NZ || i>=NR){return;} int h=i*NT*NZ+k;
  // Initialize
  double den=0.0,num=0.0;
  // Numerator....................................................
  double2 aux;
  //Radial
  aux.x=u1r[h].x-u2r[h].x; aux.y=u1r[h].y-u2r[h].y;
  num=max(num,(aux.x*aux.x+aux.y*aux.y));
  //Azimuthal
  aux.x=u1t[h].x-u2t[h].x; aux.y=u1t[h].y-u2t[h].y;
  num=max(num,(aux.x*aux.x+aux.y*aux.y));
  //Axial
  aux.x=u1z[h].x-u2z[h].x; aux.y=u1z[h].y-u2z[h].y;
  num=max(num,(aux.x*aux.x+aux.y*aux.y));
  // Denominator...................................................
  // Radial
  aux.x=u1r[h].x;          aux.y=u1r[h].y;
  den=max(den,(aux.x*aux.x+aux.y*aux.y));
  // Azimuthal
  aux.x=u1t[h].x;          aux.y=u1t[h].y;
  den=max(den,(aux.x*aux.x+aux.y*aux.y));
  // Axial
  aux.x=u1z[h].x;          aux.y=u1z[h].y;
  den=max(den,(aux.x*aux.x+aux.y*aux.y));
  // Save..........................................................
  d1[h]=num;              d2[h]=den;
} // END KERNEL FOR MEASURE CORECTION.........................................../

// KERNEL FOR RHS ADDITION (1ST STEP OF CORRECTOR)............................../
static __global__ void addRHS_k(double2 *rhsr,double2 *rhst,double2 *rhsz,
                                double2 *rhwr,double2 *rhwt,double2 *rhwz){
  // Index................................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; int i=blockIdx.y; 
  // Stopper
  if(k>=NT*NZ || i>=NR){return;} int h=i*NT*NZ+k;
  // Addittion............................................................
  // Plus
  rhsr[h].x=d_im*rhsr[h].x + (1.0-d_im)*rhwr[h].x;
  rhsr[h].y=d_im*rhsr[h].y + (1.0-d_im)*rhwr[h].y;
  // Minus
  rhst[h].x=d_im*rhst[h].x + (1.0-d_im)*rhwt[h].x;
  rhst[h].y=d_im*rhst[h].y + (1.0-d_im)*rhwt[h].y;
  // Axial
  rhsz[h].x=d_im*rhsz[h].x + (1.0-d_im)*rhwz[h].x;
  rhsz[h].y=d_im*rhsz[h].y + (1.0-d_im)*rhwz[h].y;
}// END KERNEL FOR RHS ADDITION................................................./


// KERNEL FOR MODE ZERO ZERO..................................................../
static __global__ void modeZZ_k(double2 *ur,double2 *ut,double2 *uz){
   // Index & Stopper
   int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=NR){return;} int h=i*NT*NZ;
   // Set Values
   ur[h].x=0.0; ur[h].y=0.0; ut[h].y=0.0; uz[h].y=0.0;
}// END KERNEL FOR MODE ZERO ZERO.............................................../


/*+++++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++++*/
static void modeZZ(vfield u); 
static void predictor(vfield u,vfield rhs,vfield uw,vfield rhsw,vfield ui);
static void corrector(vfield u,vfield rhs,vfield uw,vfield rhsw);
static void measucorr(double *err,double *d1,double *d2,vfield u1,vfield u2);

/*+++++++++++++++++++++++++++++++++ WRAPPERS ++++++++++++++++++++++++++++++++++*/
// MODE ZERO ZERO.............................................................../
static void modeZZ(vfield u){
   // Dimensions
   dim3 block,grid; block.x=block_size; grid.x=(NR+block.x-1)/block.x;
   // Kernel
   modeZZ_k<<<grid,block>>>(u.r,u.t,u.z); return;
}// END MODE ZERO ZERO........................................................../

/*+++++++++++++++++++++++++++++++ MAIN FUNCTIONS ++++++++++++++++++++++++++++++*/
// SET INTEGRATOR.............................................................../
void setInt(void){
    return;
}// END OF SET INTEGRATOR......................................................./

// DESTROY INTEGRATOR.........................................................../
void IntDestroy(void){
    return;
}// END OF DESTROY INTEGRATOR.................................................../

// MAIN FUNCTION TO INTEGRATE.................................................../
void integrate(vfield u,vfield rhs){
  // Initialize variables..................................................
  vfield uw, rhsw, ui;                    size_t size_p=NR*NT*NZ*sizeof(double2);
  CHECK_CUDART(cudaMalloc(&uw.r,size_p)); CHECK_CUDART(cudaMalloc(&ui.r,size_p));
  CHECK_CUDART(cudaMalloc(&uw.t,size_p)); CHECK_CUDART(cudaMalloc(&ui.t,size_p));
  CHECK_CUDART(cudaMalloc(&uw.z,size_p)); CHECK_CUDART(cudaMalloc(&ui.z,size_p));
  CHECK_CUDART(cudaMalloc(&rhsw.r,size_p));
  CHECK_CUDART(cudaMalloc(&rhsw.t,size_p)); double *d1,*d2;
  CHECK_CUDART(cudaMalloc(&rhsw.z,size_p)); size_p=NR*NT*NZ*sizeof(double);
  CHECK_CUDART(cudaMalloc(&d1,size_p));   CHECK_CUDART(cudaMalloc(&d2,size_p));
  // Pre-time step
  double t=0.0,t_frc=0.0,t_qcr=0.0,t_mnp=0.0;  int c_t=0,ite;  
  double *err=(double *)malloc(2*sizeof(double));
  // Loop on time steps.....................................................
  while(c_t<nsteps){
   // Nonlinear
   nonlinear(u,rhs);   
   // Predictor
   predictor(u,rhs,uw,rhsw,ui);
   // Initialize iteration
   err[0]=1e98; err[1]=1e99;  ite=0;
   // Loop on iterations
   while(ite<maxit && err[0]>tol){ite+=1; 
     // Nonlinear until u converge it
     nonlinear(ui,rhs);           
     // Corrector until u converge it
     corrector(u,rhs,uw,rhsw); 
     // Check convergence 
     measucorr(err,d1,d2,ui,u);
     // Copy intermediate velocity
     copyVfield(ui,u);
   }// end loop on iterations
   // Update times to the current completed time step
   c_t++; t+=dt; t_frc+=dt; t_qcr+=dt; t_mnp+=dt;
 
   // Output friction file
   if(t_frc>=dt_frc){
     // Preliminar location for the error and step verbose
     printf("\n err=%e\n",err[0]);     printf(" Step %d of %d\n", c_t,nsteps);
     // Call the friction file writer set the timer to 0
     wrt_frc(t,u.z);   t_frc=0.0;
   }// end if to write friction file

   // Output qcross and ucross file
   if(t_qcr>=dt_qcr){
     // Call the qcross file writer set the timer to 0
     wrt_qcr(u,t);   wrt_ucr(u,t);  t_qcr=0.0;
   }// end if to write qcross file

   // Output mean profile file
   if(t_mnp>=dt_mnp){
     // Call the mean profile file writer set timer to 0
     wrt_mpr(u,t); t_mnp=0.0;
   }// end if to write mean profile file

  }// End loop on time steps................................................
  // Write the h5 file
  // sprintf(Nme_h5,"fields_%08d.h5",c_t);   writeH5file(u,t,Nme_h5);
  // Post time step
  cudaFree(uw.r);     cudaFree(uw.t);     cudaFree(uw.z);   cudaFree(d1);
  cudaFree(ui.r);     cudaFree(ui.t);     cudaFree(ui.z);   cudaFree(d2);
  cudaFree(rhsw.r);   cudaFree(rhsw.t);   cudaFree(rhsw.z); free(err);  return;
}// END MAIN FUNCTION TO INTEGRATE............................................./


// PREDICTOR.................................................................../
static void predictor(vfield u,vfield rhs,vfield uw,vfield rhsw,vfield ui){
   //1. Couple ur,uth to u+,u- and udur,uduth to u+,u-
   decoupleForward(u);      decoupleForward(rhs);
   //2. Copy u and rhs to uw, rhsw
   copyVfield(uw,u);        copyVfield(rhsw,rhs);  
   //3. Add to rhs the laplacian of the last time step velocity
   addLaplacian(rhs,uw);        
   //4. Couple udu+,udu- to udur,uduth
   decoupleBackward(rhs);   
   //5. Compute the divergence of rhs (and save it to uth)
   divergence(u.t,rhs);     double2 bcon; bcon.x=0.0; bcon.y=0.0;
   //6. Solve for pressure (and save it in uz)
   write_bcon(u.t,bcon);    LUsolve_p(u.z,u.t); 
   //7. Compute gradient of pressure and add it to rhs
   addGrad(rhs,u.z);        
   //8. Couple rhs forward to +/-
   decoupleForward(rhs);   
   //9. Solve for u+,u-,uz
   write_bcon(rhs.r,bcon);  LUsolve_u(ui.r,rhs.r,0); 
   write_bcon(rhs.t,bcon);  LUsolve_u(ui.t,rhs.t,1);
   write_bcon(rhs.z,bcon);  LUsolve_u(ui.z,rhs.z,2); 
   //10.Correct BCon 
   corBCon(ui);             
   //11.Couple u back to r,th
   decoupleBackward(ui);    
   //12.Mode 0,0
   modeZZ(ui);         
   //13.Adjust the flux
   adj_flux(ui.z);       return;
} // END OF PREDICTOR........................................................../


// CORRECTOR.................................................................../
static void corrector(vfield u,vfield rhs,vfield uw,vfield rhsw){
   //1. Couple udur,uduth to u+,u-
   decoupleForward(rhs);
   //2. Construct the new rhs=d_im*rhsNew + (1-d_im)*rhsOld
   dim3 block,grid; block.x=block_size; 
   grid.x=(NT*NZ+block.x-1)/block.x;   grid.y=NR;
   addRHS_k<<<grid,block>>>(rhs.r,rhs.t,rhs.z,rhsw.r,rhsw.t,rhsw.z);
   //3. Add to rhs the laplacian of the last time step velocity
   addLaplacian(rhs,uw);
   //4. Couple udu+,udu- to udur,uduth
   decoupleBackward(rhs);
   //5. Compute the divergence of rhs (and save it to uth)
   divergence(u.t,rhs);     double2 bcon; bcon.x=0.0; bcon.y=0.0;
   //6. Solve for pressure (and save it in uz)
   write_bcon(u.t,bcon);    LUsolve_p(u.z,u.t);
   //7. Compute gradient of pressure and add it to rhs
   addGrad(rhs,u.z);
   //8. Couple rhs forward to +/-
   decoupleForward(rhs);
   //9. Solve for u+,u-,uz
   write_bcon(rhs.r,bcon);  LUsolve_u(u.r,rhs.r,0);
   write_bcon(rhs.t,bcon);  LUsolve_u(u.t,rhs.t,1);
   write_bcon(rhs.z,bcon);  LUsolve_u(u.z,rhs.z,2);
   //10.Correct BCon
   corBCon(u);
   //11.Couple u back to r,th
   decoupleBackward(u);
   //12.Mode 0,0
   modeZZ(u); 
   //13.Adjust the flux
   adj_flux(u.z);          return;
} // END OF CORRECTOR........................................................../


// MEASURE CORRECTION........................................................../
static void measucorr(double *err,double *d1,double *d2,vfield u1,vfield u2){
  // Save last error to the second position of err
  err[1]=err[0]; 
  // Save in d for even index the numerator, for odd the denominator
  dim3 block,grid; block.x=block_size; 
  grid.x=(NT*NZ+block.x-1)/block.x;   grid.y=NR;
  mcor_k<<<grid,block>>>(d1,d2,u1.r,u1.t,u1.z,u2.r,u2.t,u2.z);
  // Identify the location of the maximum numerator
  iden_max(err,d1,d2);                          return;
} // END OF MEASURE CORRECTION................................................../

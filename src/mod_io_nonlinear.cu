#include"head.h"
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to perform input/output & nonlinear operations
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 24/04/2022
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*++++++++++++++++++++++++++++++++++ KERNELS ++++++++++++++++++++++++++++++++++*/
// KERNEL TO COPY THE BASE FLOW VELOCITY......................................../
static __global__ void base_k(double2 *uz,const double *r){
  // Index
  int i=threadIdx.x+blockDim.x*blockIdx.x; 
  // Stopper
  if(i>=NR){return;} 
  // Copy value
  uz[i*NT*NZ].x=1.0-r[i]*r[i];
}// END KERNEL TO COPY THE BASE FLOW............................................/

// KERNEL TO WRITE THE INITIAL PERTURBATION...................................../
static __global__ void init_per(double *upr,double *upt,const double *rad){
  // Indexes of thread.........................................
  int k=blockIdx.x*blockDim.x + threadIdx.x;  int i=blockIdx.y;
  // azimuthal + axial wavenumber + global
  int it=k/(2*NZP), l=k%(2*NZP), h=i*2*NZP*NTP+k; 
  // Stopper
  if(l>=2*NZP || it>=NTP || i>=NR){return;} 
  // Azimuthal and axial position..............................
  double th=(double(it)/double(NTP))*LT, z=double(l)/double(2*NZP-2); 
  double ux, ur, ut, r=rad[i];
  // Compute the exp function
  ux=sin(0.5*PI2*z);  ux=ux*ux;     ux=exp(-10.0*ux);
  // Compute ur and ut 
  ur=(1.0-r*r)*(1.0-r*r);           ut=ur-4.0*r*r*(1.0-r*r);
  // Finish computing ur,ut
  ur=ur*ux*sin(th);                 ut=ut*ux*cos(th); 
  // Copy to vector
  upr[h]=A_P*ur;                    upt[h]=A_P*ut;
}// END KERNEL TO WRITE THE INITIAL PERTURBATION................................/

// KERNEL TO WRITE A NEW DATA TO FRICTION VECTOR................................/
static __global__ void wrt_frc_k(double *vec_frc,double t,double2 *uz,
                                           double *dr0,double* dr1,int cnt_frc){
   // Index & Stopper
   int thrd=threadIdx.x+blockDim.x*blockIdx.x; if(thrd>0){return;}
   // Compute the centerline velocity
   double uc=0.0;  for(int i=0;i<iw+1;i++){uc+=uz[i*NT*NZ].x*dr0[i];}
   // Compute derivative in the wall
   double du=0.0;  for(int i=0;i<sten;i++){du+=uz[(NR-sten+i)*NT*NZ].x*dr1[i];}
   // Friction velocity
   du=sqrt(fabs(du)/Re);
   // Write it together with the time in vec_frc
   vec_frc[3*cnt_frc]=t;   vec_frc[1+3*cnt_frc]=uc;  vec_frc[2+3*cnt_frc]=du;
}// END KERNEL TO WRITE A NEW DATA TO FRICTION VECTOR.........................../

// KERNEL 0 OF Q CROSS FILE...................................................../
static __global__ void q0kernel_k(double *urK,double *utK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % global index
  if(k>=2*NTP*NZP || i>=NR){return;}  int h=i*(2*NTP*NZP)+k;
  // Read velocity
  double ur=urK[h], ut=utK[h];
  // Save to urK=ur²+uth²
  urK[h]=ur*ur+ut*ut;
}// END KERNEL 0 OF Q CROSS FILE................................................/

// KERNEL 0 OF U CROSS FILE...................................................../
static __global__ void u0kernel_k(double *uzK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper % global index
  if(k>=2*NTP*NZP || i>=NR){return;}  int h=i*(2*NTP*NZP)+k;
  // Read velocity
  double uz=uzK[h];
  // Save to urK=ur²+uth²
  uzK[h]=uz*uz;
}// END KERNEL 0 OF U CROSS FILE................................................/

// KERNEL 1 OF Q CROSS FILE.......(Simpson Method)............................../
static __global__ void q1kernel_k(double *utK,double *urK){
  // Indexes of thread
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  int i=blockIdx.y*blockDim.y+threadIdx.y;
  // Stopper
  if(k>=2*NZP*NTP || i>=NR){return;}  
  // azimuthal + axial wavenumber + ih
  int it=k/(2*NZP), l=k%(2*NZP), ih=i*(2*NZP*NTP);
  // Initialize variables
  double fa,fab,fb,Int;                     int ha,h,hb;
  // Case u r in it==0
  if(it<1){         ha=ih+(NTP-1)*2*NZP+l;  hb=ih+(it+1)*2*NZP+l;}
  // Case u r in it == NTP-1
  else if(it>NTP-2){ha=ih+(it-1)*2*NZP+l;   hb=ih+(0)*2*NZP+l;}
  // Case
  else{             ha=ih+(it-1)*2*NZP+l;   hb=ih+(it+1)*2*NZP+l;}
  // Read values
  h=ih+it*2*NZP+l;  fa=urK[ha]; fab=urK[h]; fb=urK[hb];
  // Int
  Int=0.5*(1.0/(3.0*double(NTP)))*(fa+4.0*fab+fb);
  // Index to write the value and write it:
  h=l+i*2*NZP+it*(NR*2*NZP);    utK[h]=Int;  
}// END KERNEL 1 OF Q CROSS FILE................................................/

// KERNEL TO WRITE A NEW DATA LINE TO VEC_QCR ................................../
static __global__ void wrt_qcr_k(double *vec_qcr,double time,double *qin,
                                                             int cnt_qcr){
  // Index of kernel
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  // Stopper & index
  if(k>=2*NZP-1){return;} int i=cnt_qcr*(2*NZP-1)+k;
  // Time thread
  if(k<1){vec_qcr[i]=time;return;}
  // Other threads
  else{   vec_qcr[i]=qin[k-1];return;}
}// END KERNEL TO WRITE A NEW DATA LINE TO VEC_QCR............................../

// KERNEL TO WRITE A NEW DATA LINE TO VEC_MNP ................................../
static __global__ void wrt_mnp_k(double *vec_mnp,double time,double2 *uz,
                                                              int cnt_mnp){ 
  // Index of kernel
  int k=blockIdx.x*blockDim.x+threadIdx.x;
  // Stopper & index
  if(k>=NR+1){return;} int i=cnt_mnp*(NR+1)+k; 
  // Time thread
  if(k<1){vec_mnp[i]=time;return;}
  // Other threads
  else{double u=uz[(k-1)*NT*NZ].x;  vec_mnp[i]=u;return;}
}// END KERNEL TO WRITE A NEW DATA LINE TO VEC_MNP............................../










/*+++++++++++++++++++++++++++ NONLINEAR KERNELS ++++++++++++++++++++++++++++++*/
// KERNEL OF CALC NONLINEAR..................................................../
static __global__ void calcNonlinear_k(double* urK,double* utK,double* uzK,
                                     double* drurK,double* drutK,double* druzK,
                                     double* dturK,double* dtutK,double* dtuzK,
                                     double* dzurK,double* dzutK,double* dzuzK,
                                     double* r){
    // Indexes of thread
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    int i=blockIdx.y*blockDim.y+threadIdx.y;
    // Stopper % index
    if(k>=2*NTP*NZP || i>=NR){return;}  int h=i*(2*NTP*NZP)+k;
     // Write velocity & and radius
     double ur=urK[h], ut=utK[h], uz=uzK[h], rad=r[i];
     // Write dudr
     double drur=drurK[h], drut=drutK[h], druz=druzK[h];
     // Write dudt
     double dtur=dturK[h], dtut=dtutK[h], dtuz=dtuzK[h];
     // Write dudz
     double dzur=dzurK[h], dzut=dzutK[h], dzuz=dzuzK[h];
     // Udu
     double udu_r, udu_t, udu_z;
     // udu_r
     udu_r = ur*drur + ut*dtur + uz*dzur - ut*(ut/rad);
     // udu_t
     udu_t = ur*drut + ut*dtut + uz*dzut + ur*(ut/rad);
     // udu_z
     udu_z = ur*druz + ut*dtuz + uz*dzuz;
     // Write
     drurK[h]=-udu_r;    drutK[h]=-udu_t;    druzK[h]=-udu_z;
}// END KERNEL OF CALC NONLINEAR.............................................../

// KERNEL OF PADDING FORWARD.................................................../
static __global__ void padForward_k(double2 *pad,double2 *u){
    // Indexes
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int k = h%NZP; int j = h/(NTP*NZP);  int i = (h-j*NTP*NZP)/NZP;
    // Stopper
    if(h<NR*NTP*NZP){
        // Auxiliary functions
        double factor=1.0; double2 aux;   aux.x=0.0;    aux.y=0.0;
        if((i<NT/2 || i>NT-1) && k<NZ){
            // TH indices
            int ip=i<NT/2 ? i : i-NT/2;
            // Z indices
            int kp=k; int h =j*NT*NZ+ip*NZ+kp;
            // Write to aux
            aux=u[h]; aux.x*=factor; aux.y*=factor;
        }
    // Write to pad
    int hp=j*NTP*NZP+i*NZP+k;  pad[hp]=aux;
    }
}// END KERNEL OF FORWARD PADDING............................................../

// KERNEL OF PADDING BACKWARD................................................../
static __global__ void padBackward_k(double2 *u,double2 *pad){
    // Index of interest
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int k = h%NZ; int j = h/(NT*NZ); int i = (h-j*NT*NZ)/NZ;
    // Continue only if you comply with conditions
    if(j<NR & i<NT & k<NZ){
        // Auxiliary
        double factor=1.0;  double2 aux;
        // TH indices
        int ip=i<NT/2 ? i : NTP + (i-NT);
        // Z indices
        int kp=k;  int hp=j*NTP*NZP+ip*NZP+kp; int h =j*NT*NZ+i*NZ+k;
        // Write to aux
        aux=pad[hp]; aux.x*=factor; aux.y*=factor;
        // Auxiliary function
        if(i==NT/2 || k==NZ-1){ aux.x=0.0; aux.y=0.0;}
        // Write to velocity
        u[h]=aux;
    }
}// END KERNEL OF BACKWARD PADDING............................................./







/*+++++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++++*/
// Variables for all files
FILE *fp;
// Friction file
static double  *dr0,*dr1,*vec_frc; static int cnt_frc; FILE *ptr_frc; 
// Q cross file
static double *qin,*qaux,*vec_qcr; static int cnt_qcr; FILE *ptr_qcr;
// U cross file
static double *vec_ucr;            static int cnt_ucr; FILE *ptr_ucr;
// Mean profile file
static double *vec_mnp;            static int cnt_mnp; FILE *ptr_mnp;
// Stuff from old mod_nonlinear
static vfield u_pad, dur_p, dut_p, duz_p; static double *r;
static void calcNonlinear(vfield u,vfield dur,vfield dut,vfield duz);

/*+++++++++++++++++++++++++++++++++ WRAPPER ++++++++++++++++++++++++++++++++++*/
// WRAPPER OF CALC NONLINEAR.................................................../
static void calcNonlinear(vfield u,vfield dur,vfield dut,vfield duz){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(2*NTP*NZP+block.x-1)/block.x;  grid.y=NR;
    // Call Kernel
    calcNonlinear_k<<<grid,block>>>
    ((double*)u.r,  (double*)u.t,  (double*)u.z
    ,(double*)dur.r,(double*)dur.t,(double*)dur.z
    ,(double*)dut.r,(double*)dut.t,(double*)dut.z
    ,(double*)duz.r,(double*)duz.t,(double*)duz.z,r); return;
}// END WRAPPER OF CALC NONLINEAR............................................../









/*++++++++++++++++++++++++++++++++ MAIN FUNCTIONS +++++++++++++++++++++++++++++*/
// SETTER OF IN OUT VARIABLES.................................................../
void setIO(double *r_h){
  // Initial field........................................
  // Allocate r
  CHECK_CUDART(cudaMalloc(&r ,NR*sizeof(double)));
  CHECK_CUDART(cudaMemcpy(r,r_h,NR*sizeof(double),cudaMemcpyHostToDevice));
  // Friction file........................................
  // Compute dr0_h (interpolator to r=0)
  size_t size=(iw+1)*sizeof(double);
  double *dr0_h=(double *)malloc(size);     get_dr0(dr0_h,r_h);
  // Allocate dr0
  CHECK_CUDART(cudaMalloc(&dr0 ,size));
  CHECK_CUDART(cudaMemcpy(dr0,dr0_h,size,cudaMemcpyHostToDevice));
  free(dr0_h);                       size=3*dc_frc*sizeof(double);
  // Allocate dr1 (weights for derivative in r=R)  
  CHECK_CUDART(cudaMalloc(&dr1,sten*sizeof(double)));      get_dr1(dr1);
  // Allocate memory for vec_frc
  CHECK_CUDART(cudaMalloc(&vec_frc,size));
  // Open file
  ptr_frc = fopen("./io_friction.txt","w");
  // Initialize the counter
  cnt_frc=0;
  // Q crossSc file......................................
  // Define auxiliary vector for th integration
  size=NTP*sizeof(double);
  double *qaux_h=(double *)malloc(size); for(int i=0;i<NTP;i++){qaux_h[i]=1.0;}
  // Copy it to device
  CHECK_CUDART(cudaMalloc(&qaux,size));
  CHECK_CUDART(cudaMemcpy(qaux,qaux_h,size,cudaMemcpyHostToDevice));
  free(qaux_h);                size=(2*NZP-1)*dc_qcr*sizeof(double);
  // Vector of q integrated
  CHECK_CUDART(cudaMalloc(&qin,2*NZP*sizeof(double)));
  // Allocate memory for vec_qcr and vec_ucr
  CHECK_CUDART(cudaMalloc(&vec_qcr,size));
  CHECK_CUDART(cudaMalloc(&vec_ucr,size));
  // Open file
  ptr_qcr = fopen("./io_qcross.txt","w");
  ptr_ucr = fopen("./io_ucross.txt","w");
  // Initialize the counter
  cnt_qcr=0;
  // Write the first row with locations in z
  double *zfine=(double *)malloc((2*NZP-1)*sizeof(double));  zfine[0]=0.0;
  for(int i=0;i<2*NZP-2;i++){zfine[i+1]=LZ*double(i)/double(2*NZP-2);}
  for(int i=0;i<2*NZP-1;i++){fprintf(ptr_qcr,"%.8e  ",zfine[i]);
                             fprintf(ptr_ucr,"%.8e  ",zfine[i]);}
  fprintf(ptr_qcr,"\n"); fprintf(ptr_ucr,"\n"); free(zfine);
  // Mean profile file......................................
  // Allocate memory for vec_mnp
  size=(NR+1)*dc_mnp*sizeof(double); CHECK_CUDART(cudaMalloc(&vec_mnp,size));
  // Open file
  ptr_mnp = fopen("./io_meanpr.txt","w");
  // Initialize counter
  cnt_mnp=0;
  // Write the first row with locations in r
  fprintf(ptr_mnp,"%.8e  ",0.0);
  for(int i=0;i<NR;i++){fprintf(ptr_mnp,"%.8e  ",r_h[i]);} fprintf(ptr_mnp,"\n");
  return;
}// END SETTER OF IN OUT......................................................../

// DESTROY IN OUT.............................................................../
void IODestroy(void){
  // Memory
  cudaFree(r);
  cudaFree(dr0);  cudaFree(dr1);   cudaFree(vec_frc);  cudaFree(vec_mnp); 
  cudaFree(qin);  cudaFree(qaux);  cudaFree(vec_qcr);  cudaFree(vec_ucr);
  // Files
  fclose(ptr_frc);   fclose(ptr_qcr);  fclose(ptr_mnp);
  return;
}// END DESTROYER IN OUT......................................................../

// SET NONLINEAR.............................................................../
void setNonlinear(void){
    // Setting the size of buffers
    size_t size_pad=NR*NTP*NZP*sizeof(double2);
    // Allocate u_pad
    CHECK_CUDART(cudaMalloc(&u_pad.r,size_pad));
    CHECK_CUDART(cudaMalloc(&u_pad.t,size_pad));
    CHECK_CUDART(cudaMalloc(&u_pad.z,size_pad));
    // Allocate dur_pad
    CHECK_CUDART(cudaMalloc(&dur_p.r,size_pad));
    CHECK_CUDART(cudaMalloc(&dur_p.t,size_pad));
    CHECK_CUDART(cudaMalloc(&dur_p.z,size_pad));
    // Allocate dut_pad
    CHECK_CUDART(cudaMalloc(&dut_p.r,size_pad));
    CHECK_CUDART(cudaMalloc(&dut_p.t,size_pad));
    CHECK_CUDART(cudaMalloc(&dut_p.z,size_pad));
    // Allocate duz_pad
    CHECK_CUDART(cudaMalloc(&duz_p.r,size_pad));
    CHECK_CUDART(cudaMalloc(&duz_p.t,size_pad));
    CHECK_CUDART(cudaMalloc(&duz_p.z,size_pad));
    return;
}// END SETTER................................................................../

// DESTROY NONLINEAR............................................................/
void NonDestroy(void){
    cudaFree(u_pad.r);   cudaFree(u_pad.t);   cudaFree(u_pad.z);
    cudaFree(dur_p.r);   cudaFree(dur_p.t);   cudaFree(dur_p.z);
    cudaFree(dut_p.r);   cudaFree(dut_p.t);   cudaFree(dut_p.z);
    cudaFree(duz_p.r);   cudaFree(duz_p.t);   cudaFree(duz_p.z);  return;
}// END NONLINEAR DESTROYER...................................................../

// PAD FORWARD................................................................../
void padForward(double2 *pad,double2 *u){
    // Dimension
    dim3 grid,block; block.x=block_size;
    grid.x = (NR*NTP*NZP + block.x - 1)/block.x;
    // Call kernel
    padForward_k<<<grid,block>>>(pad,u); return;
}// END PAD FORWARD............................................................./

// PAD BACKWARD................................................................./
void padBackward(double2 *u,double2 *pad){
    // Dimension
    dim3 grid,block; block.x=block_size;
    grid.x = (NR*NTP*NZP + block.x - 1)/block.x;
    // Call kernel
    padBackward_k<<<grid,block>>>(u,pad); return;
}// END PAD BACKWARD............................................................/

// NONLINEAR MAIN FUNCTION....................................................../
void nonlinear(vfield u,vfield du){
   // 1. Compute the radial derivative
   derivr(du.r,u.r,1);       derivr(du.t,u.t,1);       derivr(du.z,u.z,0);
   // 2. Pad forward the velocity
   padForward(u_pad.r,u.r);  padForward(u_pad.t,u.t);  padForward(u_pad.z,u.z);
   // 3. Pad forward the radial derivative of the velocity
   padForward(dur_p.r,du.r); padForward(dur_p.t,du.t); padForward(dur_p.z,du.z);
   // 4. Calculate the azimuthal derivative
   derivt(du.r,u.r);         derivt(du.t,u.t);         derivt(du.z,u.z);
   padForward(dut_p.r,du.r); padForward(dut_p.t,du.t); padForward(dut_p.z,du.z);
   // 5. Calculate the axial     derivative
   derivz(du.r,u.r);         derivz(du.t,u.t);         derivz(du.z,u.z);
   padForward(duz_p.r,du.r); padForward(duz_p.t,du.t); padForward(duz_p.z,du.z);
   // 6. fftBackward of u
   fftBackward(u_pad.r);     fftBackward(u_pad.t);     fftBackward(u_pad.z);
   // 7. fftBackward of dudr
   fftBackward(dur_p.r);     fftBackward(dur_p.t);     fftBackward(dur_p.z);
   // 8. fftBackward of dudt
   fftBackward(dut_p.r);     fftBackward(dut_p.t);     fftBackward(dut_p.z);
   // 9. fftBackward of dudz
   fftBackward(duz_p.r);     fftBackward(duz_p.t);     fftBackward(duz_p.z);
   // 10.Calc Nonlinear
   calcNonlinear(u_pad,dur_p,dut_p,duz_p);
   // 11.fftForward  of dudr
   fftForward(dur_p.r);      fftForward(dur_p.t);      fftForward(dur_p.z);
   // 12.Pad Backward the rhs
   padBackward(du.r,dur_p.r);padBackward(du.t,dur_p.t);padBackward(du.z,dur_p.z);
   // 13.Add to mode 0,0 the pressure gradient
   pressGrad(du.z,u.z);      return;
}// END MAIN NONLINEAR FUNCTION................................................./












/*++++++++++++++++++++++++++++++ INITIAL FIELD ++++++++++++++++++++++++++++++++*/
// INIT FIELD.................................................................../
void initField(vfield u){
   // Read .bin files Careful they must have the same length u input
   if(restart>0){ printf("Reading initial condition\n");
    rBufferBinary((double*)u.r,"ur_s.bin",sizeof(double2),NR*NT*NZ);
    rBufferBinary((double*)u.t,"ut_s.bin",sizeof(double2),NR*NT*NZ);
    rBufferBinary((double*)u.z,"uz_s.bin",sizeof(double2),NR*NT*NZ);}
   // Usually u initialize with the optimum perturbation
   else{ printf("Preparing initial condition\n");
   // Auxiliary variables
    double2 uni; uni.x=0.0; uni.y=0.0; 
   // Set current velocity to 0
    uniformVec(u.r,uni); uniformVec(u.t,uni); uniformVec(u.z,uni);
   // Call kernel to introduce the parabolic profile
    dim3 grid,block; block.x=block_size; grid.x=(NR+block.x-1)/block.x;
    base_k<<<grid,block>>>(u.z,r); 
   // Pad forward
    padForward(u_pad.r,u.r);  padForward(u_pad.t,u.t); 
   // Write the perturbation
    grid.x=(2*NTP*NZP+block.x-1)/block.x;  grid.y=NR; 
    init_per<<<grid,block>>>((double*)u_pad.r,(double*)u_pad.t,r);
   // Do forward fft
    fftForward(u_pad.r);      fftForward(u_pad.t);   
   // Pad backward
    padBackward(u.r,u_pad.r); padBackward(u.t,u_pad.t);}  
   return;   
}// END INIT FIELD............................................................../








/*++++++++++++++++++++++++++++++ FRICTION FILE ++++++++++++++++++++++++++++++++*/
// WRITE TO FRICTION FILE......................................................./
void wrt_frc(double t,double2 *uz){
  // Call kernel to write the value inside vec_frc
  dim3 grid,block; block.x=block_size; grid.x=1; 
  wrt_frc_k<<<grid,block>>>(vec_frc,t,uz,dr0,dr1,cnt_frc);
  // Sum to the counter +1
  cnt_frc+=1; 
  // If cnt_frc==dc_frc write 
  if(cnt_frc==dc_frc){ 
    // Copy to host
    size_t size=3*dc_frc*sizeof(double);
    double *v_f_h=(double *)malloc(size);
    CHECK_CUDART(cudaMemcpy(v_f_h,vec_frc,size,cudaMemcpyDeviceToHost));
    // Write
    for(int i=0;i<dc_frc;i++){
      fprintf(ptr_frc,"%.8e  %.8e  %.8e\n",v_f_h[3*i],v_f_h[1+3*i],v_f_h[2+3*i]);
    }// end loop
    // Reset counter       
    cnt_frc=0; free(v_f_h);
  } return;
}// END WRITE TO FRICTION FILE................................................../






/*++++++++++++++++++++++++++++++++++ HDF5 FILES +++++++++++++++++++++++++++++++*/
// WRITE THE .H5 FILE.........................................................../
//void writeH5file(vfield u,double time,const char* Nme){
  // Create filename
  //hid_t file_id;                                double Lz=LZ,Lt=LT; 
  //file_id = H5Fcreate(Nme, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  // Go from spectral to physical quantities
  //padForward(u_pad.r,u.r);  padForward(u_pad.t,u.t);  padForward(u_pad.z,u.z);
  //fftBackward(u_pad.r);     fftBackward(u_pad.t);     fftBackward(u_pad.z);
  // Create velocities variables in the host
  //size_t size=NR*NTP*NZP*sizeof(double2);
  //static double2* ur=(double2*)malloc(size);
  //static double2* ut=(double2*)malloc(size);
  //static double2* uz=(double2*)malloc(size);
  // Copy the velocity from device to host
  //CHECK_CUDART(cudaMemcpy(ur,u_pad.r,size,cudaMemcpyDeviceToHost));
  //CHECK_CUDART(cudaMemcpy(ut,u_pad.t,size,cudaMemcpyDeviceToHost));
  //CHECK_CUDART(cudaMemcpy(uz,u_pad.z,size,cudaMemcpyDeviceToHost));
  // Write the velocity to file
  //hsize_t dims[3] = {NR,NTP,2*NZP};
  //H5LTmake_dataset(file_id, "/ur", 3, dims, H5T_IEEE_F64LE, ur);
  //H5LTmake_dataset(file_id, "/ut", 3, dims, H5T_IEEE_F64LE, ut);
  //H5LTmake_dataset(file_id, "/uz", 3, dims, H5T_IEEE_F64LE, uz);
  // Write radial grid
  //hsize_t dimr[1] = {NR}; 
  //static double *r_h=(double*)malloc(NR*sizeof(double));
  //CHECK_CUDART(cudaMemcpy(r_h,r,NR*sizeof(double),cudaMemcpyDeviceToHost));
  //H5LTmake_dataset(file_id, "/r",  1, dimr, H5T_IEEE_F64LE, r_h);
  // Write the time and other variables
  //hsize_t dd[1] = {1};
  //H5LTmake_dataset(file_id, "/t",  1,dd, H5T_IEEE_F64LE,&time);
  //H5LTmake_dataset(file_id, "/Lz", 1,dd, H5T_IEEE_F64LE,&Lz);
  //H5LTmake_dataset(file_id, "/Lt", 1,dd, H5T_IEEE_F64LE,&Lt);
  // Free velocities
  //free(ur);    free(ut);   free(uz);    free(r_h);
//}// END WRITE THE .H5 FILE....................................................../




/*++++++++++++++++++++++++++++++++ WRITE BUFFER +++++++++++++++++++++++++++++++*/
// WRITE BUFFER................................................................./
void wBufferBinary(double* w,const char* file,size_t elsize,size_t elements){
    // Allocate memory in host create sizes
    double* t_host=(double*)malloc(elsize*elements); size_t size=elements;
    // Copy w to host
    CHECK_CUDART(cudaMemcpy(t_host,w,elsize*elements, cudaMemcpyDeviceToHost));
    // Open file
    fp=fopen(file,"wb"); 
    if(fp==NULL){printf("\nwriting error: %s",file); exit(1);}
    // Write to file
    size_t fsize =fwrite( (unsigned char*)t_host,elsize,size,fp);
    if(fsize!=size){ printf("\nwriting error: %s",file); exit(1);}
    // Close file and free host
    fclose(fp);    free(t_host); return;
}// END BUFFER WRITER.........................................................../






/*+++++++++++++++++++++++++++++++++ READ BUFFER +++++++++++++++++++++++++++++++*/
// READ BUFFER................................................................../
void rBufferBinary(double* w,const char* file,size_t elsize,size_t elements){
    // Allocate memory in host create sizes
    double* t_host=(double*)malloc(elsize*elements); size_t size=elements;
    // Open file
    fp=fopen(file,"rb");
    if(fp==NULL){printf("\nreading error: %s",file); exit(1);}
    // Read file
    size_t fsize =fread( (unsigned char*)t_host,elsize,size,fp);
    if(fsize!=size){ printf("\nreading error: %s",file); exit(1);}
    // Close file
    fclose(fp);
    // Copy to device
    CHECK_CUDART(cudaMemcpy(w,t_host,elsize*elements,cudaMemcpyHostToDevice));
    // Free host and return
    free(t_host); return;
}// END BUFFER READER.........................................................../








/*+++++++++++++++++++++++++++++++ QCROSS FILE +++++++++++++++++++++++++++++++++*/
// WRITE TO QCROSS FILE........................................................./
void wrt_qcr(vfield u,double time){
  // Go from spectral to physical quantities
  padForward(u_pad.r,u.r);  padForward(u_pad.t,u.t);
  fftBackward(u_pad.r);     fftBackward(u_pad.t);     
  // Kernel to compute q=ur²+uth²
  // Dimensions
  dim3 grid,block; block.x=block_size;
  grid.x=(2*NTP*NZP+block.x-1)/block.x;  grid.y=NR;
  // Call kernel: q is in u_pad.r
  q0kernel_k<<<grid,block>>>((double*)u_pad.r,(double*)u_pad.t);
  // Kernel to integrate wrt th using a simpson method 
  // Call kernel: q is in u_pad.t
  q1kernel_k<<<grid,block>>>((double*)u_pad.t,(double*)u_pad.r);
  // Call function to integrate q using CUBLAS
  intq(qin,(double*)u_pad.t,qaux);
  // Call kernel to write the value inside vec_qcr
  grid.x=(2*NZP-1+block.x-1)/block.x;         grid.y=1;
  wrt_qcr_k<<<grid,block>>>(vec_qcr,time,qin,cnt_qcr);
  // Sum to the counter +1
  cnt_qcr+=1;
  // If cnt_qcr==dc_qcr write 
  if(cnt_qcr==dc_qcr){
    // Copy to host
    size_t size=(2*NZP-1)*dc_qcr*sizeof(double);
    double *v_q_h=(double *)malloc(size);
    CHECK_CUDART(cudaMemcpy(v_q_h,vec_qcr,size,cudaMemcpyDeviceToHost));
    // Write
    for(int i=0;i<dc_qcr;i++){
     for(int j=0;j<2*NZP-1;j++){
      fprintf(ptr_qcr,"%.8e  ",v_q_h[i*(2*NZP-1)+j]);
     }// end loop in j
     fprintf(ptr_qcr,"\n");
    }// end loop in i
    // Reset counter       
    cnt_qcr=0;  free(v_q_h);
  } return;
}// END QCROSS FILE............................................................./





/*+++++++++++++++++++++++++++++++ UCROSS FILE +++++++++++++++++++++++++++++++++*/
// WRITE TO UCROSS FILE........................................................./
void wrt_ucr(vfield u,double time){
  // Go from spectral to physical quantities
  padForward(u_pad.z,u.z);     fftBackward(u_pad.z);
  // Dimensions
  dim3 grid,block; block.x=block_size;
  grid.x=(2*NTP*NZP+block.x-1)/block.x;  grid.y=NR;
  // Call kernel: u is in u_pad.z
  u0kernel_k<<<grid,block>>>((double*)u_pad.z);
  // Kernel to integrate wrt th using a simpson method (u)...........
  // Call kernel: u2 is in u_pad.t
  q1kernel_k<<<grid,block>>>((double*)u_pad.t,(double*)u_pad.z);
  // Call function to integrate u2 using CUBLAS
  intq(qin,(double*)u_pad.t,qaux);
  // Call kernel to write the value inside vec_ucr
  grid.x=(2*NZP-1+block.x-1)/block.x;         grid.y=1;
  wrt_qcr_k<<<grid,block>>>(vec_ucr,time,qin,cnt_ucr);
  // Sum to the counter +1
  cnt_ucr+=1;
  // If cnt_ucr==dc_qcr write
  if(cnt_ucr==dc_qcr){
    // Copy to host q
    size_t size=(2*NZP-1)*dc_qcr*sizeof(double);
    double *v_u_h=(double *)malloc(size);
    CHECK_CUDART(cudaMemcpy(v_u_h,vec_ucr,size,cudaMemcpyDeviceToHost));
    // Write
    for(int i=0;i<dc_qcr;i++){
     for(int j=0;j<2*NZP-1;j++){
      fprintf(ptr_ucr,"%.8e  ",v_u_h[i*(2*NZP-1)+j]);
     }// end loop in j
     fprintf(ptr_ucr,"\n");
    }// end loop in i
    // Reset counter
    cnt_ucr=0;       free(v_u_h);
  } return;
}// END UCROSS FILE............................................................./






/*+++++++++++++++++++++++++++++++ MEANPR FILE +++++++++++++++++++++++++++++++++*/
// WRITE TO MEAN PROFILE......................................................../
void wrt_mpr(vfield u,double time){
    // Call kernel to write the value inside vec_mnp
    dim3 grid,block; block.x=block_size;
    grid.x=(NR+block.x)/block.x;  grid.y=1;
    // Call kernel to write to vec_mnp
    wrt_mnp_k<<<grid,block>>>(vec_mnp,time,u.z,cnt_mnp);
    // Sum to the counter +1
    cnt_mnp+=1;
    // If cnt_mnp==dc_mnp write
    if(cnt_mnp==dc_mnp){
    // Copy to host mnp
    size_t size=(NR+1)*dc_mnp*sizeof(double);
    double *v_m_h=(double *)malloc(size);
    CHECK_CUDART(cudaMemcpy(v_m_h,vec_mnp,size,cudaMemcpyDeviceToHost));
    // Write
    for(int i=0;i<dc_mnp;i++){
     for(int j=0;j<NR+1;j++){
      fprintf(ptr_mnp,"%.8e  ",v_m_h[i*(NR+1)+j]);
     }// end loop in j
     fprintf(ptr_mnp,"\n");
    }// end loop in i
    // Reset counter
    cnt_mnp=0;       free(v_m_h);
    } return;
}// END MEANPR FILE............................................................./

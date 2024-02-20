#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to calculate radial grid, integral 
//                     and derivatives.
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 27/03/2022
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*++++++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++++*/
// RADIAL DERIVATIVE KERNEL..................................................../
static __global__ void derivr_k(double2 *dur,double2 *u,const double *D,int flag){
  // Index........................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; int i=blockIdx.y;
  // Stopper
  if(k>=NT*NZ || i>=NR){return;} int h=i*NT*NZ+k;
  // Azimuthal wavenumber
  int it=k/NZ,left=max(0,i-iw),right=min(i+iw,NR-1);  it=it<NT/2 ? it:it-NT;
  // Prepare flag for parity condition even(0) odd(1)
  int ip0=abs((abs(it)%2)-flag)*sten*NR;  double2 prod,aux; double de;
  // Length
  int len=right+1-left, i0=i+iw-right; prod.x=0.0; prod.y=0.0; 
  // Loop on the columns of matrix..................................
  for(int j=0;j<len;j++){de=D[ip0+sten*i+j+i0];
     // Read value from velocity 
     aux.x=u[(left+j)*NT*NZ+k].x; aux.y=u[(left+j)*NT*NZ+k].y;
     // Sum to the product
     prod.x+=de*aux.x; prod.y+=de*aux.y; 
  }// end loop
  // Write to the value.............................................
  dur[h].x=prod.x;  dur[h].y=prod.y;
}//END KERNEL OF RADIAL DERIVATIVE............................................./


// WALL DIVERGENCE KERNEL....................................................../
static __global__ void divWall_k(double2 *div1,double2 *ur,double2 *ut,
                                              double2 *uz,const double *D){
  // Index........................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; int i=NR-1;
  // Stopper location in vector and axial wavenumber
  if(k>=NT*NZ){return;} int h=i*NT*NZ+k;   int l=k%NZ;
  // Azimuthal wavenumber
  int it=k/NZ, left=i-iw, right=NR-1;  it=it<NT/2 ? it:it-NT;
  // Wavenumbers
  double kt=(PI2/LT)*(double(it));  double kz=(PI2/LZ)*(double(l));
  // Prepare flag for parity condition even(0) odd(1)
  int ip0=abs((abs(it)%2)-1)*sten*NR;  double2 prod,aux; double de;
  // Length
  int len=right+1-left, i0=i+iw-right; prod.x=0.0; prod.y=0.0;
  // Loop on the columns of matrix..................................
  for(int j=0;j<len;j++){de=D[ip0+sten*i+j+i0];
     // Read value from velocity
     aux.x=ur[(left+j)*NT*NZ+k].x; aux.y=ur[(left+j)*NT*NZ+k].y;
     // Sum to the product
     prod.x+=de*aux.x; prod.y+=de*aux.y;
  }// end loop
  // Write the value of ur[h].......................................
  prod.x+=    ur[h].x;     prod.y+=   ur[h].y;
  // Write the value of dut/dt......................................
  prod.x+=-kt*ut[h].y;     prod.y+=kt*ut[h].x;
  // Write the value of duz/dz......................................
  prod.x+=-kz*uz[h].y;     prod.y+=kz*uz[h].x;
  // Write to the value the div.....................................
  div1[k].x=prod.x;        div1[k].y=prod.y;
}//END KERNEL OF WALL DIVERGENCE.............................................../


// WALL DIVERGENCE KERNEL....................................................../
static __global__ void div_k(double2 *div1,double2 *ur,double2 *ut,double2 *uz,
                                              const double *D,const double *r){
  // Index........................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; int i=blockIdx.y;
  // Stopper location in vector and axial wavenumber
  if(k>=NT*NZ || i>=NR){return;} int h=i*NT*NZ+k; int l=k%NZ;
  // Azimuthal wavenumber
  int it=k/NZ, left=max(0,i-iw), right=min(i+iw,NR-1);  it=it<NT/2 ? it:it-NT;
  // Wavenumbers
  double kt=(PI2/LT)*(double(it));  double kz=(PI2/LZ)*(double(l));
  // Prepare flag for parity condition even(0) odd(1)
  int ip0=abs((abs(it)%2)-1)*sten*NR;  double2 prod,aux; double de,rK=r[i];
  // Length
  int len=right+1-left, i0=i+iw-right; prod.x=0.0; prod.y=0.0;
  // Loop on the columns of matrix..................................
  for(int j=0;j<len;j++){de=D[ip0+sten*i+j+i0];
     // Read value from velocity
     aux.x=ur[(left+j)*NT*NZ+k].x; aux.y=ur[(left+j)*NT*NZ+k].y;
     // Sum to the product
     prod.x+=de*aux.x;             prod.y+=de*aux.y;
  }// end loop
  // Write the value of ur[h].......................................
  prod.x+=    ur[h].x/rK;     prod.y+=   ur[h].y/rK;
  // Write the value of dut/dt......................................
  prod.x+=-kt*ut[h].y/rK;     prod.y+=kt*ut[h].x/rK;
  // Write the value of duz/dz......................................
  prod.x+=   -kz*uz[h].y;     prod.y+=   kz*uz[h].x;
  // Write to the value the div.....................................
  div1[h].x=prod.x;           div1[h].y=prod.y;
}//END KERNEL OF WALL DIVERGENCE.............................................../


// PRESSURE GRADIENT KERNEL..................................................../
static __global__ void pressGrad_k(double2 *duz,double2 *uz,const double *D){
  // Index..................               
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=NR){return;} int h=i*NT*NZ; 
  // Index for derivative wrt of r
  int iE=NR-1, left=iE-iw, right=NR-1, i0=iE+iw-right, len=right+1-left;
  // Initialize the derivative
  double de,du=0.0;
  // Loop to compute the derivative at the wall
  for(int j=0;j<len;j++){de=D[sten*iE+j+i0]; du+=de*uz[(left+j)*NT*NZ].x;} 
  // Write the value
  duz[h].x-=(2.0/Re)*du;
}// END PRESSURE GRADIENT KERNEL.............................................../


// ADD GRADIENT TO RHS KERNEL................................................../
static __global__ void addGrad_k(double2 *rhsr,double2 *rhst,double2 *rhsz,
                                   double2* p,const double *D,const double *r){
  // Index........................................................
  int k=blockIdx.x*blockDim.x+threadIdx.x; int i=blockIdx.y;
  // Stopper location in vector and axial wavenumber
  if(k>=NT*NZ || i>=NR){return;} int h=i*NT*NZ+k; int l=k%NZ;
  // Azimuthal wavenumber
  int it=k/NZ, left=max(0,i-iw), right=min(i+iw,NR-1);  it=it<NT/2 ? it:it-NT;
  // Wavenumbers
  double kt=(PI2/LT)*(double(it));  double kz=(PI2/LZ)*(double(l));
  // Prepare flag for parity condition even(0) odd(1)
  int ip0=abs((abs(it)%2)-0)*sten*NR;  double2 prod,aux; double de,rK=r[i];
  // Length
  int len=right+1-left, i0=i+iw-right; prod.x=0.0; prod.y=0.0;
  // Loop on the columns of matrix..................................
  for(int j=0;j<len;j++){de=D[ip0+sten*i+j+i0];
     // Read value from velocity
     aux.x=p[(left+j)*NT*NZ+k].x; aux.y=p[(left+j)*NT*NZ+k].y;
     // Sum to the product
     prod.x+=de*aux.x;            prod.y+=de*aux.y;
  }// end loop
  // Add the components to each rhs.................................
  // Radial
  rhsr[h].x-=prod.x;              rhsr[h].y-=prod.y;
  // Azimuthal
  rhst[h].x-=-kt*p[h].y/rK;       rhst[h].y-=kt*p[h].x/rK;
  // Axial
  rhsz[h].x-=-kz*p[h].y;          rhsz[h].y-=kz*p[h].x;
}//END ADD GRADIENT TO RHS KERNEL............................................../


/*++++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++++*/
static double *D,*r; 
static void fd_weights(double *c,double *rau,int i,int left,int right,int m);


/*++++++++++++++++++++++++++++++++++ WRAPPERS ++++++++++++++++++++++++++++++++*/
// WRAPPER OF RADIAL DERIVATIVE................................................/
void derivr(double2 *dur,double2 *u,int flag){
  // Dimensions
  dim3 grid,block; block.x=block_size;
  grid.x=(NT*NZ+block.x-1)/block.x; grid.y=NR;
  // Kernel
  derivr_k<<<grid,block>>>(dur,u,D,flag); return;
}// END WRAPPER OF RADIAL DERIVATIVE.........................................../

// WRAPPER OF DIVERGENCE AT THE WALL.........................................../
void divWall(double2 *div1,vfield u){
  // Dimensions
  dim3 grid,block; block.x=block_size; grid.x=(NT*NZ+block.x-1)/block.x;
  // Kernel
  divWall_k<<<grid,block>>>(div1,u.r,u.t,u.z,D); return;
}// END WRAPPER OF DIVERGENCE AT THE WALL....................................../

// WRAPPER OF DIVERGENCE......................................................./
void divergence(double2 *div1,vfield u){
  // Dimensions
  dim3 grid,block; block.x=block_size; 
  grid.x=(NT*NZ+block.x-1)/block.x;  grid.y=NR;
  // Kernel
  div_k<<<grid,block>>>(div1,u.r,u.t,u.z,D,r); 
  return;
}// END WRAPPER OF DIVERGENCE................................................../

// WRAPPER OF PRESSURE GRADIENT................................................/
void pressGrad(double2 *duz,double2* uz){
  // Dimensions
  dim3 grid,block; block.x=block_size; grid.x=(NR+block.x-1)/block.x;
  // Kernel
  pressGrad_k<<<grid,block>>>(duz,uz,D); return;
}// END WRAPPER OF PRESSURE GRADIENT............................................/

// WRAPPER TO ADD THE GRADIENT TO THE RHS....................................../
void addGrad(vfield rhs,double2 *p){
  // Grid dimension
  dim3 block,grid; block.x=block_size;
  grid.x=(NT*NZ+block.x-1)/block.x; grid.y=NR;
  // Call kernel
  addGrad_k<<<grid,block>>>(rhs.r,rhs.t,rhs.z,p,D,r);  return;
}// END WRAPPER TO ADD THE GRADIENTTO THE RHS................................../


/*+++++++++++++++++++++++++++++++ MAIN FUNCTIONS ++++++++++++++++++++++++++++++*/
// INITIALIZE RADIAL GRID......................................................./
void init_fd(double *r_h,double *Lu_h,double *Lp_h,double* rdr_h){
 // Initialize auxiliary variables..........................................
  int ir,hD=sten*NR;  
  double *rau  =(double *)malloc((NR+iw)*sizeof(double));
  double *r_i  =(double *)malloc((NR+1 )*sizeof(double));
  double *D_h  =(double *)malloc(2*sten*NR*sizeof(double));
  double *D2_h =(double *)malloc(2*sten*NR*sizeof(double));
  double *c    =(double *)malloc(sten*sten*sizeof(double));
 // Initialize derivative matrices. They go column->row->even/odd
  for(int i=0;i<2*hD;i++){D_h[i]=0.0;   D2_h[i]=0.0;}
  for(int i=0;i<NR;i++){rdr_h[i]=0.0;}
 // 0.0 Calculate grid as in nsPipe........................................
  int N_p=NR+int(sqrt(double(NR))); double dr;
  for(int i=N_p-NR;i<N_p;i++){ir=N_p-i-1;
     r_h[i-N_p+NR]=0.5*(1.0+cos(0.5*PI2*(double(ir)/double(N_p))));}
  // Stuff they do to make it denser to the center
  for(int i=0;i<10;i++){dr=1.5*r_h[0]-0.5*r_h[1];
     for(int j=0;j<NR;j++){r_h[j]=r_h[j]*(1.0+dr)-dr;}}
 // 0.1. Auxiliary grids
  for(int i=0;i<NR;i++){r_i[i+1]=r_h[i];} r_i[0]=0.0;
  for(int i=0;i<NR+iw;i++){if(i<iw){rau[i]=-r_h[iw-i-1];}else{rau[i]=r_h[i-iw];}}
// 1 Calculate derivative weights.........................................
  int m=2; int left,left2,right,len,i0,coun=iw;
 // 1.1 Loop on rows 
  for(int i=0;i<NR;i++){
   // Index
   left=i; right=min(iw+iw+i,NR+iw-1); len=1+right-left;
   // Actual weights
   fd_weights(c,rau,i,left,right,m);
   // 1.2 Write to vectors the boundary condition
   if(i<iw){
    // 1.3 Loop on j (rows with len<sten will have patched 0's after)
    for(int j=0;j<len;j++){
      // 1.4 Condition
      if(j<coun){ir=coun-j-1;
      D_h[ ir+sten*i]+=c[  len+j]; D_h[ hD+ir+sten*i]-=c[  len+j];
      D2_h[ir+sten*i]+=c[2*len+j]; D2_h[hD+ir+sten*i]-=c[2*len+j];
      }
      else{ir=j-coun;
      D_h[ ir+sten*i]+=c[  len+j]; D_h[ hD+ir+sten*i]+=c[  len+j];
      D2_h[ir+sten*i]+=c[2*len+j]; D2_h[hD+ir+sten*i]+=c[2*len+j];
      } // end else        
     } // end loop on j
     coun-=1;
   } // end if on B.Con
   // 1.5 Write the rest of the rows
   else{i0=sten-len;
    // 1.6 Loop on j (rows with len<sten will have patched 0's before)
    for(int j=0;j<len;j++){ir=i0+j;
      D_h[ ir+sten*i]+=c[  len+j]; D_h[ hD+ir+sten*i]+=c[  len+j];
      D2_h[ir+sten*i]+=c[2*len+j]; D2_h[hD+ir+sten*i]+=c[2*len+j];
    } // end loop on j
   } // end else for other points
  } // end loop on i....................................................
 // 2. Calculate the weights for integration............................
 // rdr: is the weights for:          1/(piR²)·int_0^R·int_0^2pi()rdrdth
 double e1,e2; coun=iw;
 // 2.1 Loop on rows
  for(int i=0;i<NR;i++){
   // Index
   left=i; right=min(iw+iw+i,NR+iw-1); len=1+right-left; left2=max(0,i-iw);
   // Actual weights
   fd_weights(c,rau,i,left,right,len-1); i0=0; e1=1.0; e2=1.0;
   // 2.2 Write to vectors the boundary condition
   if(i<iw){i0=coun;   
    // 2.3 Loop on j
    for(int j=0;j<coun;j++){ir=coun+coun-1-j;  
       //2.4 Loop on derivatives 
       for(int k=0;k<len;k++){c[ir+k*len]+=c[j+k*len];}
    } // end loop on j
    coun-=1; 
   }// end if condition
   // 2.5 Loop on derivatives
   for(int k=0;k<len+1;k++){
      e1=e1*(r_i[min(i+2,NR)]-r_i[i+1])/(double(k+1));
      e2=e2*(r_i[i]          -r_i[i+1])/(double(k+1)); 
      // 2.6 Loop on integral weights
      for(int j=0;j<len-i0;j++){ir=i0+j;
         rdr_h[left2+j]+=(e1-e2)*r_h[i]*c[ir+k*len];
      } // end loop on j
   } // end loop on k
  } // end loop on i.....................................................
// 3. Prepare the Radial part of the laplacian...........................
 for(int i=0;i<NR;i++){
  if(i<NR-1){
   // Loop on columns
   for(int j=0;j<sten;j++){ir = j+sten*i;
     // Write the radial part of the laplacian for u and p
     Lu_h[ir]=D2_h[ir]+D_h[ir]/r_h[i]; 
     Lu_h[hD+ir]=D2_h[hD+ir]+D_h[hD+ir]/r_h[i];
     Lp_h[ir]=D2_h[ir]+D_h[ir]/r_h[i]; 
     Lp_h[hD+ir]=D2_h[hD+ir]+D_h[hD+ir]/r_h[i];
   }// end loop on j
  }// end if on i<NR-1
  else{
   // Loop on columns
   for(int j=0;j<sten;j++){ir = j+sten*i;
     // Write the boundary condition in the laplacian itself   
     if(j<sten-1){Lu_h[ir]=0.0; Lu_h[hD+ir]=0.0;}
     else{        Lu_h[ir]=1.0; Lu_h[hD+ir]=1.0;}
     Lp_h[ir]=D_h[ir];          Lp_h[hD+ir]=D_h[hD+ir];
   }// end loop on j
  }// end if so i=NR-1
 }// end loop on i.......................................................
 // 4. Locate the variables in device....................................
 // Malloc
 CHECK_CUDART(cudaMalloc((void**)&r, NR*sizeof(double)));
 CHECK_CUDART(cudaMalloc((void**)&D , 2*sten*NR*sizeof(double)));
 // CudaMemcpy
 CHECK_CUDART(cudaMemcpy(r,r_h,NR*sizeof(double),cudaMemcpyHostToDevice));
 CHECK_CUDART(cudaMemcpy(D,D_h,2*sten*NR*sizeof(double),cudaMemcpyHostToDevice));
 // 5. Free host.........................................................
 free(rau);   free(r_i);  free(c); free(D_h); free(D2_h); return;
} // END INITIAL RADIAL......................................................../


// FUNCTION TO COMPUTE DR0 (INTERPOLATE TO r=0)................................/
void get_dr0(double *dr0,double *r){
    // Initialize auxiliary variables
    int mn,n=iw+1,m=0; double c1,c2,c3,c4,c5,x[n],x0=0.0; 
    // Initialize c vector
    double *c  =(double *)malloc(n*sizeof(double));
    for(int i=0;i<n;i++){c[i]=0.0;} c[0]=1.0; c1=1.0;
    // Initialize x vector
    for(int i=0;i<n;  i++){x[i]=r[i];} c4=x[0]-x0;
    // Loop on points on x
    for(int i=1;i<n;  i++){
      mn=min(i,m);  c2=1.0; c5=c4; c4=x[i]-x0;
      // Loop on auxiliary points
      for(int j=0;j<i;j++){
        c3=x[i]-x[j]; c2=c2*c3;
        // if j==i-1
        if(j==i-1){
          // Loop on derivatives
          for(int k=mn;k>0;k--){
            c[i+n*k]=c1*(double(k)*c[i-1+n*(k-1)]-c5*c[i-1+n*k])/c2;
          } // end loop on k
          c[i]=-c1*c5*c[i-1]/c2;
        } // end if j==i-1
        // Loop on derivatives
        for(int k=mn;k>0;k--){
           c[j+n*k]=(c4*c[j+n*k]-double(k)*c[j+(k-1)*n])/c3;
        } // end loop on k
        c[j]=c4*c[j]/c3;
      } // end loop on j
      c1=c2;
    } // end loop on i
   // Copy to dr0
   for(int i=0;i<iw+1;i++){dr0[i]=c[i];} free(c);
   // Free host 
   return;
}// END FUNCTION TO COMPUTE DR0................................................/



// FUNCTION TO COMPUTE DR1 (DERIVATIVE AT r=R)................................./
void get_dr1(double* dr1){
   // Copy
CHECK_CUDART(cudaMemcpy(dr1,D+(NR-1)*sten,sten*sizeof(double),cudaMemcpyDeviceToDevice));
}// END FUNCTION TO COMPUTE DR1................................................/



/*++++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++++*/
// A.F.0: Compute the m derivatives at x0 with the use of a lagrange 
//        interpolation using the points between left and right
static void fd_weights(double* c,double* rau,int i0,int left,int right,int m){
 // Initialize auxiliary variables
 int mn,n=1+right-left; double c1,c2,c3,c4,c5,x[n],x0=rau[i0+iw];
 // Initialize c vector
 for(int i=0;i<sten*sten;i++){c[i]=0.0;} c[0]=1.0; c1=1.0;
 // Initialize x vector
 for(int i=0;i<n;  i++){x[i]=rau[i+i0];} c4=x[0]-x0;
 // Loop on points on x
 for(int i=1;i<n;  i++){
   mn=min(i,m);  c2=1.0; c5=c4; c4=x[i]-x0;
   // Loop on auxiliary points
   for(int j=0;j<i;j++){
     c3=x[i]-x[j]; c2=c2*c3;
     // if j==i-1
     if(j==i-1){
       // Loop on derivatives
       for(int k=mn;k>0;k--){
         c[i+n*k]=c1*(double(k)*c[i-1+n*(k-1)]-c5*c[i-1+n*k])/c2;
       } // end loop on k
       c[i]=-c1*c5*c[i-1]/c2;
     } // end if j==i-1
     // Loop on derivatives
     for(int k=mn;k>0;k--){
        c[j+n*k]=(c4*c[j+n*k]-double(k)*c[j+(k-1)*n])/c3;
     } // end loop on k
     c[j]=c4*c[j]/c3;
   } // end loop on j
   c1=c2;
 } // end loop on i
 return;
} // end function fd_weights..................................................../

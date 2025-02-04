#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to remesh the grid
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 27/03/2022
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// KERNEL TO REMESH......................................................../
static __global__ void rmsh_k(double2 *u,double2 *uo,double *RM,int flag){
  // Indexes of thread
  int k_o =blockIdx.x*blockDim.x+threadIdx.x; 
  int h_or=blockIdx.y*blockDim.y+threadIdx.y; 
  // Stopper
  if(k_o>=NT*NZ || h_or>=Nrd){return;} 
  // Index of the old 
  int h_oz=k_o%NZ, it=k_o/NZ, h_ot=it; h_ot=h_ot<NT/2 ? h_ot:h_ot-NT; 
  // Stopper in case you are out of the bounds of the new mesh
  if(h_ot>=Ntd/2 || h_ot<-Ntd/2 || h_oz>=Nzd){return;}
  // Even/odd
  int ip0=abs((abs(h_ot)%2)-flag)*NR*Nrd;
  // Create the azimuthal index and actual index in the field
  if(h_ot<0){h_ot=h_ot+Ntd;}  int h=h_or*Ntd*Nzd + h_ot*Nzd + h_oz; 
  // Initialize
  double2 du; du.x=0.0; du.y=0.0; int ia; double rm;
  // Interpolate using matrix RM
  for(int j=0;j<NR;j++){rm=RM[h_or*NR+j+ip0]; ia=j*NZ*NT+k_o; 
	  du.x+=rm*uo[ia].x;              du.y+=rm*uo[ia].y;}
  // Record value
  u[h].x=du.x; u[h].y=du.y;
}// END KERNEL TO REMESH.................................................../



// KERNEL FOR MODE ZERO ZERO.............................................../
static __global__ void modeZZ_k(double2 *ur,double2 *ut,double2 *uz){
   // Index & Stopper
   int i=blockIdx.x*blockDim.x+threadIdx.x; 
   if(i>=Nrd){return;} int h=i*Ntd*Nzd;
   // Set Values
   ur[h].x=0.0; ur[h].y=0.0; ut[h].y=0.0; uz[h].y=0.0;
}// END KERNEL FOR MODE ZERO ZERO........................................../


/*+++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++*/
static void wrp_rmsh(double2 *u,double2 *uold,double *RM,int flag);
static void modeZZ(vfield u);
static void fd_weights_rmsh(double* c,double* x,double x0,int n);

/*+++++++++++++++++++++++++++++ MAIN FUNCTIONS +++++++++++++++++++++++++++*/
// MAIN FUNCTION TO PERFORM THE RE-MESHING................................./
void re_mesh(vfield u){
 // 0.Initialize old field: allocate and load the previous field
 vfield uO;                size_t size_p=NR*NT*NZ*sizeof(double2);
 CHECK_CUDART(cudaMalloc(&uO.r,size_p));
 CHECK_CUDART(cudaMalloc(&uO.t,size_p));
 CHECK_CUDART(cudaMalloc(&uO.z,size_p));
 rBufferBinary((double*)uO.r,"ur_o.bin",sizeof(double2),NR*NT*NZ);
 rBufferBinary((double*)uO.t,"ut_o.bin",sizeof(double2),NR*NT*NZ);
 rBufferBinary((double*)uO.z,"uz_o.bin",sizeof(double2),NR*NT*NZ); 
 // 1.Generate old radial grid······································
 double *r_o=(double *)malloc(NR*sizeof(double));
 int N_p=NR+int(sqrt(double(NR))); double dr;  int ir;
 // Load and generate the old radial grid
 for(int i=N_p-NR;i<N_p;i++){ir=N_p-i-1;
     r_o[i-N_p+NR]=0.5*(1.0+cos(0.5*PI2*(double(ir)/double(N_p))));}
 // Stuff they do to make it denser to the center
 for(int i=0;i<10;i++){dr=1.5*r_o[0]-0.5*r_o[1];
     for(int j=0;j<NR;j++){r_o[j]=r_o[j]*(1.0+dr)-dr;}}
 // Extended original grid
 double *rau  =(double *)malloc((NR+iw)*sizeof(double));
 for(int i=0;i<NR+iw;i++){if(i<iw){rau[i]=-r_o[iw-i-1];}else{rau[i]=r_o[i-iw];}}
 // 2.Generate new radial grid······································
 double *r_h=(double *)malloc(Nrd*sizeof(double));
 N_p=Nrd+int(sqrt(double(Nrd))); 
 // Load and generate the old radial grid
 for(int i=N_p-Nrd;i<N_p;i++){ir=N_p-i-1;
     r_h[i-N_p+Nrd]=0.5*(1.0+cos(0.5*PI2*(double(ir)/double(N_p))));}
 // Stuff they do to make it denser to the center
 for(int i=0;i<10;i++){dr=1.5*r_h[0]-0.5*r_h[1];
     for(int j=0;j<Nrd;j++){r_h[j]=r_h[j]*(1.0+dr)-dr;}}
 // 3.Try interpolation weights
 double *RM_h=(double *)malloc(2*NR*Nrd*sizeof(double));   
 for(int i=0;i<2*NR*Nrd;i++){RM_h[i]=0.0;}
 RM_h[NR*Nrd-1]=1.0; RM_h[2*NR*Nrd-1]=1.0;
 // 4.Loop on rows.................................................
 double x0; int jleft,jright,n,jaux,jcol;
 // 4.0 Initialize stuff and enter the loop
 for(int i=0;i<Nrd-1;i++){x0=r_h[i]; 
  // 4.1 Identify the first point to consider
  for(int j=0;j<NR+iw;j++){if(rau[j]>=x0){jleft=j;break;}}
  // 4.2 Identify the index of interest
  jleft=jleft-iw; jright=min(NR+iw-1,jleft+2*iw-1); n=jright-jleft+1;
  // 4.3 Create the vectors of interest
  double *x=(double *)malloc(n*sizeof(double));
  double *c=(double *)malloc(n*sizeof(double));
  for(int j=0;j<n;j++){x[j]=rau[jleft+j]; c[j]=0.0;}
  // 4.4 Call the function to compute c
  fd_weights_rmsh(c,x,x0,n); jleft=jleft-iw; jaux=jleft;
  // 4.5 Loop on columns
  for(int j=0;j<n;j++){
    // Case too close to the boundary
    if(jaux<0){jcol=iw-j-1;  jaux+=1;
      RM_h[i*NR+jcol]+=c[j]; RM_h[NR*Nrd+i*NR+jcol]-=c[j];
    }
    // Otherwise
    else{jcol=jleft+j;
      RM_h[i*NR+jcol]+=c[j]; RM_h[NR*Nrd+i*NR+jcol]+=c[j];
    }
  }// end loop on columns
  // 4.X Destroy
  free(x); free(c); 
 }// end loop on rows
 // 5.Copy to Device and call the kernel
 double *RM;  CHECK_CUDART(cudaMalloc(&RM,2*NR*Nrd*sizeof(double))); 
 CHECK_CUDART(cudaMemcpy(RM,RM_h,2*NR*Nrd*sizeof(double),cudaMemcpyHostToDevice));
 // 6.REMESH radial velocity
 wrp_rmsh(u.r,uO.r,RM,1);  wrp_rmsh(u.t,uO.t,RM,1);  wrp_rmsh(u.z,uO.z,RM,0);
 // 7.Adjust mode 0,0
 modeZZ(u);
 // Destroy CUDA
 cudaFree(RM); cudaFree(uO.r); cudaFree(uO.t); cudaFree(uO.z);
 // Destroy locally
 free(RM_h); free(rau); free(r_h); free(r_o); 
 return;
}// END MAIN FUNCTION TO PERFORM THE RE-MESHING............................/

/*++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++*/
// WRAPPER OF ADD STD....................................................../
static void wrp_rmsh(double2 *u,double2 *uold,double *RM,int flag){
    // Dimensions
    dim3 grid,block; block.x=block_size;
    grid.x=(NT*NZ+block.x-1)/block.x;  grid.y=Nrd;
    // Call Kernel
    rmsh_k<<<grid,block>>>(u,uold,RM,flag); return;
}// END WRAPPER OF ADD STD................................................./

// MODE ZERO ZERO........................................................../
static void modeZZ(vfield u){
   // Dimensions
   dim3 block,grid; block.x=block_size; grid.x=(Nrd+block.x-1)/block.x;
   // Kernel
   modeZZ_k<<<grid,block>>>(u.r,u.t,u.z); return;
}// END MODE ZERO ZERO...................................................../


/*++++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++++*/
// A.F.0: Compute the m derivatives at x0 with the use of a lagrange
//        interpolation using the points between left and right
static void fd_weights_rmsh(double* c,double* x,double x0,int n){
 // Initialize auxiliary variables
 int mn,m=0; double c1,c2,c3,c4,c5;
 // Initialize c vector
 c[0]=1.0; c1=1.0;
 // Initialize x vector
 c4=x[0]-x0;
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

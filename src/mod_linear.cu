#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to solve linear systems and compute Lap
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 11/04/2022
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*+++++++++++++++++++++++++++++ INTERNAL FUNCTIONS +++++++++++++++++++++++++++*/
static double *Lp, *Lu, *r;

/*++++++++++++++++++++++++++++++++++ KERNELS +++++++++++++++++++++++++++++++++*/
// KERNEL TO PERFORM Lap*u MULTIPLICATION....................................../
static __global__ void addLap_k(double2* rhs,double2* u, const double *LHS,
                                             const double *r,int flag){
   // Index...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x; int i=blockIdx.y;
   // Stopper and initialize variables
   if(k>=NT*NZ || i>=NR){return;} double kz,kt; int h=i*NT*NZ+k; 
   // Prepare flag for parity condition (u+(0),u-(1))(1) (uz(2),p(3))(0)
   int fp=abs(flag-3)/2;
   // Azimuthal wavenumber and axial wavenumber
   int it=k/NZ,       l=k%NZ;        it=it<NT/2 ? it:it-NT;
   // Prepare flag for parity condition even(0) odd(1)
   int ip0=abs((abs(it)%2)-fp)*sten*NR;
   // Axial and azimuthal wavenumbers 
   kz=(PI2/LZ)*(double(l));          kt=(PI2/LT)*(double(it));
   // Radial part of the laplacian.........................................
   double2 prod,aux; double de; int left=max(0,i-iw),right=min(i+iw,NR-1);
   // Length
   int len=right+1-left, i0=i+iw-right; prod.x=0.0; prod.y=0.0;
   // Loop on the columns of matrix
   for(int j=0;j<len;j++){de=LHS[ip0+sten*i+j+i0];
     // Read value from velocity 
     aux.x=u[(left+j)*NT*NZ+k].x; aux.y=u[(left+j)*NT*NZ+k].y;
     // Sum to the product
     prod.x+=de*aux.x;            prod.y+=de*aux.y;
   }// end loop
   // Full laplacian........................................................
   aux=u[h];  double r2=r[i]*r[i]; de=-kz*kz-kt*kt/r2;
   // U+
   if(flag==0)     {de+=-(1.0/r2)-(2.0*kt/r2);}
   // U-
   else if(flag==1){de+=-(1.0/r2)+(2.0*kt/r2);}
   // Add to prod
   prod.x+=de*aux.x;        prod.y+=de*aux.y; 
   // Multiply the laplacian
   prod.x*=(1.0-d_im)/Re;   prod.y*=(1.0-d_im)/Re;
   // Final expresion.......................................................
   prod.x+=(1.0/dt)*aux.x;  prod.y+=(1.0/dt)*aux.y;
   // Add it to the rhs
   rhs[h].x+=prod.x;      rhs[h].y+=prod.y;
}// END KERNEL TO ADD LAPLACIAN................................................/


// KERNEL TO PERFORM LU DECOMPOSITION of UP..................................../
static __global__ void LUsolve_up_k(double2 *sol,double2 *RHS,const double *LHS,
                                              const double *r){
   // Index...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x;
   // Stopper and initialize variables
   if(k>=NT*NZ){return;} double kz,kt; 
   // Prepare flag for parity condition (u+(0),u-(1))(1) (uz(2),p(3))(0)
   // Azimuthal wavenumber and axial wavenumber
   int it=k/NZ,       l=k%NZ;        it=it<NT/2 ? it:it-NT;     
   // Prepare flag for parity condition even(0) odd(1)
   int ip0=abs((abs(it)%2)-1)*sten*NR;
   // Axial and azimuthal wavenumbers 
   kz=(PI2/LZ)*(double(l));          kt=(PI2/LT)*(double(it));

   // LU solve............................................................
   // Initialize auxiliary vectors
   double L[1+iw],U[(1+iw)*NR]; double2 x[NR];
   // Initialize auxiliary variables
   int left=0,hleft,right=iw,coun=0,coun2=0,i0=0; double XM,r2=r[0]*r[0];

   // 0 First row of U matrix............................................
   int i=0;   L[0]=1.0;   x[0].x=RHS[k].x;   x[0].y=RHS[k].y; 
   // First column......................................
   // Matrix
   XM=LHS[ip0+sten*i+i0+coun];  
   // Main diagonal add axial+azimuthal derivatives
   XM+=-(1.0/r2)-(2.0*kt/r2)-kz*kz-kt*kt/r2;    XM=(1.0/dt)-(d_im/Re)*XM;
   // Initial value on U
   U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   // Rest of columns...................................
   for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun]; 
    // Case u r not in the main diagonal
    XM=-(d_im/Re)*XM;
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   }// end loop on j
   
   // 1 Construct U matrix................................................
   for(int i=1;i<NR-1;i++){r2=r[i]*r[i];
   // Index as in matrix formulation + L vector (represents L matrix)
   left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
   // Initialize L and Save values of the RHS
   L[i-left]=1.0;    
   // Save values of RHS in x
   x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y; 
   // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . . . . . . . 
   // Loop on j
   for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];  
     // Differentiate between pressure(3) and velocity(<3) and Bcon!
     XM=-(d_im/Re)*XM;       
     // Write X to L
     L[j]=XM;  coun+=1;   
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L 
     L[j]=L[j]/U[(j+left)*(1+iw)]; 
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . . . . . . . .
  // First column...............................................
  // Initial value on X
  XM=LHS[ip0+sten*i+i0+coun];
  // Case u r in the main diagonal and not in Bcon!
  // U+
  XM+=-(1.0/r2)-(2.0*kt/r2)-kz*kz-kt*kt/r2;  XM=(1.0/dt)-(d_im/Re)*XM;
  // Initial value on U
  U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
  // Substract U by L*U
  if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[0+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+0+coun2-h];}
  } // end if 
  // Rest of columns............................................
  for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun];
    // Case u r not in the main diagonal and not in Bcon!
    XM=-(d_im/Re)*XM;
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if 
   }// end loop on j
  }// end loop on i.......................................................... 
  
  // 2 Finish construction of U (last row)...................................
  i=NR-1; r2=r[i]*r[i];
  // Index as in matrix formulation + L vector (represents L matrix)
  left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
  // Initialize L and Save values of the RHS
  L[i-left]=1.0;
  // Save values of RHS in x
  x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y;
  // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . ..
  // Loop on j
  for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];
     // Write X to L
     L[j]=XM;  coun+=1;
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L
     L[j]=L[j]/U[(j+left)*(1+iw)];
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . ..
  for(int j=0;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun];
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if
  }// end loop on j.........................................................

  // 3 Backward substitution to obtain sol...................................
  // 3.0 Invert U for the first i
  i=NR-1;
   x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;
   x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
  // Continue inverting U
  for(int i=NR-2;i>-1;i--){
    // Indexes
    right=min(i+iw,NR-1);
    // 3.1 Invert U
    for(int j=1;j<right-i+1;j++){
          x[i].x-=U[i*(iw+1)+j]*x[i+j].x;     
          x[i].y-=U[i*(iw+1)+j]*x[i+j].y;
    }
    // 3.2 Obtain x and write it to the correct solution position
    x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;  
    x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
 } // end loop on i.......................................................... 
}// END KERNEL LUSOLVE of UP.................................................../


// KERNEL TO PERFORM LU DECOMPOSITION of UM..................................../
static __global__ void LUsolve_um_k(double2 *sol,double2 *RHS,const double *LHS,
                                              const double *r){
   // Index...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x;
   // Stopper and initialize variables
   if(k>=NT*NZ){return;}    double kz,kt; 
   // Prepare flag for parity condition (u+(0),u-(1))(1) (uz(2),p(3))(0)
   // Azimuthal wavenumber and axial wavenumber
   int it=k/NZ,       l=k%NZ;        it=it<NT/2 ? it:it-NT;     
   // Prepare flag for parity condition even(0) odd(1)
   int ip0=abs((abs(it)%2)-1)*sten*NR;
   // Axial and azimuthal wavenumbers 
   kz=(PI2/LZ)*(double(l));          kt=(PI2/LT)*(double(it));

   // LU solve............................................................
   // Initialize auxiliary vectors
   double L[1+iw],U[(1+iw)*NR]; double2 x[NR];
   // Initialize auxiliary variables
   int left=0,hleft,right=iw,coun=0,coun2=0,i0=0; double XM,r2=r[0]*r[0];

   // 0 First row of U matrix............................................
   int i=0;   L[0]=1.0;   x[0].x=RHS[k].x;   x[0].y=RHS[k].y; 
   // First column......................................
   // Matrix
   XM=LHS[ip0+sten*i+i0+coun];  
   // Main diagonal of U-
   XM+=-(1.0/r2)+(2.0*kt/r2)-kz*kz-kt*kt/r2;  XM = (1.0/dt)-(d_im/Re)*XM;
   // Initial value on U
   U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   // Rest of columns...................................
   for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun]; 
    // Case u r not in the main diagonal
    XM=-(d_im/Re)*XM;
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   }// end loop on j
   
   // 1 Construct U matrix................................................
   for(int i=1;i<NR-1;i++){r2=r[i]*r[i];
   // Index as in matrix formulation + L vector (represents L matrix)
   left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
   // Initialize L and Save values of the RHS
   L[i-left]=1.0;    
   // Save values of RHS in x
   x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y; 
   // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . . . . . . . 
   // Loop on j
   for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];  
     // Differentiate between pressure(3) and velocity(<3) and Bcon!
     XM=-(d_im/Re)*XM;       
     // Write X to L
     L[j]=XM;  coun+=1;   
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L 
     L[j]=L[j]/U[(j+left)*(1+iw)]; 
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . . . . . . . .
  // First column...............................................
  // Initial value on X
  XM=LHS[ip0+sten*i+i0+coun];
  // Case u r in the main diagonal
  // U-
  XM+=-(1.0/r2)+(2.0*kt/r2)-kz*kz-kt*kt/r2;    XM = (1.0/dt)-(d_im/Re)*XM;
  // Initial value on U
  U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
  // Substract U by L*U
  if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[0+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+0+coun2-h];}
  } // end if 
  // Rest of columns............................................
  for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun];
    // Case u r not in the main diagonal and not in Bcon!
    XM=-(d_im/Re)*XM;
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if 
   }// end loop on j
  }// end loop on i.......................................................... 
  
  // 2 Finish construction of U (last row)...................................
  i=NR-1; r2=r[i]*r[i];
  // Index as in matrix formulation + L vector (represents L matrix)
  left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
  // Initialize L and Save values of the RHS
  L[i-left]=1.0;
  // Save values of RHS in x
  x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y;
  // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . ..
  // Loop on j
  for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];
     // Write X to L
     L[j]=XM;  coun+=1;
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L
     L[j]=L[j]/U[(j+left)*(1+iw)];
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . ..
  for(int j=0;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun];
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if
  }// end loop on j.........................................................

  // 3 Backward substitution to obtain sol...................................
  // 3.0 Invert U for the first i
  i=NR-1;
   x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;
   x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
  // Continue inverting U
  for(int i=NR-2;i>-1;i--){
    // Indexes
    right=min(i+iw,NR-1);
    // 3.1 Invert U
    for(int j=1;j<right-i+1;j++){
          x[i].x-=U[i*(iw+1)+j]*x[i+j].x;     
          x[i].y-=U[i*(iw+1)+j]*x[i+j].y;
    }
    // 3.2 Obtain x and write it to the correct solution position
    x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;  
    x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
 } // end loop on i..........................................................
}// END KERNEL LUSOLVE of UM.................................................../


// KERNEL TO PERFORM LU DECOMPOSITION of UZ..................................../
static __global__ void LUsolve_uz_k(double2 *sol,double2 *RHS,const double *LHS,
                                              const double *r){
   // Index...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x;
   // Stopper and initialize variables
   if(k>=NT*NZ){return;} double kz,kt; 
   // Prepare flag for parity condition (u+(0),u-(1))(1) (uz(2),p(3))(0)
   // Azimuthal wavenumber and axial wavenumber
   int it=k/NZ,       l=k%NZ;        it=it<NT/2 ? it:it-NT;     
   // Prepare flag for parity condition even(0) odd(1)
   int ip0=abs((abs(it)%2)-0)*sten*NR;
   // Axial and azimuthal wavenumbers 
   kz=(PI2/LZ)*(double(l));          kt=(PI2/LT)*(double(it));

   // LU solve............................................................
   // Initialize auxiliary vectors
   double L[1+iw],U[(1+iw)*NR]; double2 x[NR];
   // Initialize auxiliary variables
   int left=0,hleft,right=iw,coun=0,coun2=0,i0=0; double XM,r2=r[0]*r[0];

   // 0 First row of U matrix............................................
   int i=0;   L[0]=1.0;   x[0].x=RHS[k].x;   x[0].y=RHS[k].y; 
   // First column......................................
   // Matrix
   XM=LHS[ip0+sten*i+i0+coun];  
   // Main diagonal
   // Uz
   XM+=-kz*kz-kt*kt/r2; XM=(1.0/dt)-(d_im/Re)*XM;
   // Initial value on U
   U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   // Rest of columns...................................
   for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun]; 
    // Case u r not in the main diagonal
    XM=-(d_im/Re)*XM;
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   }// end loop on j
   
   // 1 Construct U matrix................................................
   for(int i=1;i<NR-1;i++){r2=r[i]*r[i];
   // Index as in matrix formulation + L vector (represents L matrix)
   left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
   // Initialize L and Save values of the RHS
   L[i-left]=1.0;    
   // Save values of RHS in x
   x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y; 
   // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . . . . . . . 
   // Loop on j
   for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];  
     // Differentiate between pressure(3) and velocity(<3) and Bcon!
     XM=-(d_im/Re)*XM;       
     // Write X to L
     L[j]=XM;  coun+=1;   
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L 
     L[j]=L[j]/U[(j+left)*(1+iw)]; 
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . . . . . . . .
  // First column...............................................
  // Initial value on X
  XM=LHS[ip0+sten*i+i0+coun];
  // Case u r in the main diagonal and not in Bcon!
  XM+=-kz*kz-kt*kt/r2; XM=(1.0/dt)-(d_im/Re)*XM;
  // Initial value on U
  U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
  // Substract U by L*U
  if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[0+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+0+coun2-h];}
  } // end if 
  // Rest of columns............................................
  for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun];
    // Case u r not in the main diagonal and not in Bcon!
    XM=-(d_im/Re)*XM;
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if 
   }// end loop on j
  }// end loop on i.......................................................... 
  
  // 2 Finish construction of U (last row)...................................
  i=NR-1; r2=r[i]*r[i];
  // Index as in matrix formulation + L vector (represents L matrix)
  left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
  // Initialize L and Save values of the RHS
  L[i-left]=1.0;
  // Save values of RHS in x
  x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y;
  // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . ..
  // Loop on j
  for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];
     // Write X to L
     L[j]=XM;  coun+=1;
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L
     L[j]=L[j]/U[(j+left)*(1+iw)];
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . ..
  for(int j=0;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun]; 
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if
  }// end loop on j.........................................................

  // 3 Backward substitution to obtain sol...................................
  // 3.0 Invert U for the first i
  i=NR-1;
   x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;
   x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
  // Continue inverting U
  for(int i=NR-2;i>-1;i--){
    // Indexes
    right=min(i+iw,NR-1);
    // 3.1 Invert U
    for(int j=1;j<right-i+1;j++){
          x[i].x-=U[i*(iw+1)+j]*x[i+j].x;     
          x[i].y-=U[i*(iw+1)+j]*x[i+j].y;
    }
    // 3.2 Obtain x and write it to the correct solution position
    x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;  
    x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
 } // end loop on i.......................................................... 
}// END KERNEL LUSOLVE of UZ.................................................../


// KERNEL TO PERFORM LU DECOMPOSITION of P...................................../
static __global__ void LUsolve_p_k(double2 *sol,double2 *RHS,const double *LHS,
                                              const double *r){
   // Index...............................................................
   int k=blockIdx.x*blockDim.x+threadIdx.x;
   // Stopper and initialize variables
   if(k>=NT*NZ){return;} double kz,kt; 
   // Prepare flag for parity condition (u+(0),u-(1))(1) (uz(2),p(3))(0)
   // Azimuthal wavenumber and axial wavenumber
   int it=k/NZ,       l=k%NZ;        it=it<NT/2 ? it:it-NT;     
   // Prepare flag for parity condition even(0) odd(1)
   int ip0=abs((abs(it)%2)-0)*sten*NR;
   // Axial and azimuthal wavenumbers 
   kz=(PI2/LZ)*(double(l));          kt=(PI2/LT)*(double(it));

   // LU solve............................................................
   // Initialize auxiliary vectors
   double L[1+iw],U[(1+iw)*NR]; double2 x[NR];
   // Initialize auxiliary variables
   int left=0,hleft,right=iw,coun=0,coun2=0,i0=0; double XM,r2=r[0]*r[0];

   // 0 First row of U matrix............................................
   int i=0;   L[0]=1.0;   x[0].x=RHS[k].x;   x[0].y=RHS[k].y; 
   // First column......................................
   // Matrix
   XM=LHS[ip0+sten*i+i0+coun];  
   // Main diagonal
   // P
   XM+=-kz*kz-kt*kt/r2;
   // Initial value on U
   U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   // Rest of columns...................................
   for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun]; 
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
   }// end loop on j
   
   // 1 Construct U matrix................................................
   for(int i=1;i<NR-1;i++){r2=r[i]*r[i];
   // Index as in matrix formulation + L vector (represents L matrix)
   left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
   // Initialize L and Save values of the RHS
   L[i-left]=1.0;    
   // Save values of RHS in x
   x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y; 
   // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . . . . . . . 
   // Loop on j
   for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];  
     // Write X to L
     L[j]=XM;  coun+=1;   
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L 
     L[j]=L[j]/U[(j+left)*(1+iw)]; 
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . . . . . . . .
  // First column...............................................
  // Initial value on X
  XM=LHS[ip0+sten*i+i0+coun];
  // Case u r in the main diagonal and not in Bcon!
  // P
  XM+=-kz*kz-kt*kt/r2;
  // Initial value on U
  U[0+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
  // Substract U by L*U
  if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[0+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+0+coun2-h];}
  } // end if 
  // Rest of columns............................................
  for(int j=1;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun];
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if 
   }// end loop on j
  }// end loop on i.......................................................... 
  
  // 2 Finish construction of U (last row)...................................
  i=NR-1; r2=r[i]*r[i];
  // Index as in matrix formulation + L vector (represents L matrix)
  left=max(0,i-iw); right=min(i+iw,NR-1); coun=0;  i0=i+iw-right;
  // Initialize L and Save values of the RHS
  L[i-left]=1.0;
  // Save values of RHS in x
  x[i].x=RHS[i*NT*NZ+k].x;   x[i].y=RHS[i*NT*NZ+k].y;
  // 1.1 Columns only on L. . . . . . . . . . . . . . . . . . ..
  // Loop on j
  for(int j=0;j<i-left;j++){
     // Write the LHS value to X (not in the main diagonal of LHS)
     XM=LHS[ip0+j+i0+sten*i];
     // Careful with mode 00 of pressure
     if(abs(l)<1 && abs(it)<1){XM=0.0;}
     // Write X to L
     L[j]=XM;  coun+=1;
     // Discount to L the L*U of earlier iterations
     if(j>0){for(int h=0;h<j;h++){L[j]-=L[h]*U[(1+iw)*(h+left)+j-h];}}
     // Compute L
     L[j]=L[j]/U[(j+left)*(1+iw)];
     // Directly discount it from x
     x[i].x-=L[j]*x[j+left].x;    x[i].y-=L[j]*x[j+left].y;
  } // end loop on j
  coun2=coun;  // auxiliary index
  // 1.2 Columns only on U . . . . . . . . . . . . . . . . . ..
  for(int j=0;j<right-i+1;j++){
    // Initial value on X
    XM=LHS[ip0+sten*i+i0+coun];
    // Careful with mode 00 of pressure
    if(abs(l)<1 && abs(it)<1){XM=0.0;  if(j==0){XM=1.0;} } 
    // Initial value on U
    U[j+(1+iw)*i]=XM; hleft=max(0,coun-iw); coun+=1;
    // Substract U by L*U
    if(hleft<i-left){for(int h=hleft;h<i-left;h++){
          U[j+(1+iw)*i]-=L[h]*U[(1+iw)*(h+left)+j+coun2-h];}
    } // end if
  }// end loop on j.........................................................

  // 3 Backward substitution to obtain sol...................................
  // 3.0 Invert U for the first i
  i=NR-1;
   x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;
   x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
  // Continue inverting U
  for(int i=NR-2;i>-1;i--){
    // Indexes
    right=min(i+iw,NR-1);
    // 3.1 Invert U
    for(int j=1;j<right-i+1;j++){
          x[i].x-=U[i*(iw+1)+j]*x[i+j].x;     
          x[i].y-=U[i*(iw+1)+j]*x[i+j].y;
    }
    // 3.2 Obtain x and write it to the correct solution position
    x[i].x=x[i].x/U[i*(iw+1)];   sol[i*NT*NZ+k].x=x[i].x;  
    x[i].y=x[i].y/U[i*(iw+1)];   sol[i*NT*NZ+k].y=x[i].y;
 } // end loop on i.......................................................... 
}// END KERNEL LUSOLVE........................................................./



/*++++++++++++++++++++++++++++++++++ WRAPPER +++++++++++++++++++++++++++++++++*/
// WRAPPER TO ADD THE LAPLACIAN TO THE RHS...................................../
void addLaplacian(vfield rhs,vfield u){
  // Grid dimension
  dim3 block,grid; block.x=block_size; 
  grid.x=(NT*NZ+block.x-1)/block.x; grid.y=NR;
  // Call kernels
  addLap_k<<<grid,block>>>(rhs.r,u.r,Lu,r,0);
  addLap_k<<<grid,block>>>(rhs.t,u.t,Lu,r,1);
  addLap_k<<<grid,block>>>(rhs.z,u.z,Lu,r,2);  
  return;
}// END WRAPPER TO ADD THE LAPLACIAN.........................................../

// WRAPPER TO LUSOLVE VELOCITY................................................./
void LUsolve_u(double2 *sol,double2 *RHS,int flag){
   // Grid dimension 
   dim3 block,grid; block.x=block_size; grid.x=(NT*NZ+block.x-1)/block.x;
   // Call Kernel
        if(flag==0){LUsolve_up_k<<<grid,block>>>(sol,RHS,Lu,r);}
   else if(flag==1){LUsolve_um_k<<<grid,block>>>(sol,RHS,Lu,r);}
   else if(flag==2){LUsolve_uz_k<<<grid,block>>>(sol,RHS,Lu,r);}  return;
}// END WRAPPER LUSOLVE VELOCITY.............................................../

// WRAPPER TO LUSOLVE PRESSURE................................................./
void LUsolve_p(double2 *sol,double2 *RHS){
   // Grid dimension
   dim3 block,grid; block.x=block_size; grid.x=(NT*NZ+block.x-1)/block.x;
   // Call Kernel
   LUsolve_p_k<<<grid,block>>>(sol,RHS,Lp,r);   return;
}// END WRAPPER LUSOLVE PRESSURE.............................................../



/*++++++++++++++++++++++++++++++ MAIN FUNCTION +++++++++++++++++++++++++++++++*/
// SETTER OF LUSOLVE.........................................................../
void setLinear(double *r_h,double *Lu_h,double *Lp_h){
   // Malloc
   CHECK_CUDART(cudaMalloc(&r ,NR*sizeof(double)));
   CHECK_CUDART(cudaMalloc(&Lu,2*sten*NR*sizeof(double)));
   CHECK_CUDART(cudaMalloc(&Lp,2*sten*NR*sizeof(double)));
   // CudaMemcpy
   CHECK_CUDART(cudaMemcpy(r,r_h,NR*sizeof(double),cudaMemcpyHostToDevice));
   CHECK_CUDART(cudaMemcpy(Lu,Lu_h,2*sten*NR*sizeof(double),cudaMemcpyHostToDevice));
   CHECK_CUDART(cudaMemcpy(Lp,Lp_h,2*sten*NR*sizeof(double),cudaMemcpyHostToDevice)); 
   return;
}// END MAIN FUNCTION OF SETTER................................................./


// DESTROYER OF LUSOLVE........................................................./
void LinearDestroy(void){
   return;
}// END DESTROYER OF LU........................................................./




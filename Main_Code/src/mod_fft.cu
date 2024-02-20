#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to perform ffts
// AUTHOR: ----------- Alberto Vela Mart√n
// DATE:   ----------- 27/03/2022
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

/*++++++++++++++++++++++++++++ INTERNAL FUNCTIONS ++++++++++++++++++++++++++++*/
static cufftHandle fft2_r2c_zp;
static cufftHandle fft2_c2r_zp;
// Checker of fft in cuda
void cufftCheck(cufftResult error, const char* function ){
    if(error != CUFFT_SUCCESS){
       printf("\n error  %s : %d \n", function, error); exit(1);
    }
    return;
}

/*++++++++++++++++++++++++++++++ MAIN FUNCTIONS ++++++++++++++++++++++++++++++*/
// INITIALIZE THE FFT..........................................................
void setFft(size_p sizes){
    // Size of the points after aliasing
    int nzp[2]={NTP,2*NZP-2};
    // Planner of the fft 
cufftCheck(cufftPlanMany(&fft2_r2c_zp,2,nzp,NULL,1,0,NULL,1,0,CUFFT_D2Z,NR),"ALLOCATE_FFT3_R2C_ZP");
cufftCheck(cufftPlanMany(&fft2_c2r_zp,2,nzp,NULL,1,0,NULL,1,0,CUFFT_Z2D,NR),"ALLOCATE_FFT3_C2R_ZP");
    // Allocate the auxiliary vector
    return;
}//END SETTER...................................................................

// DESTROY THE FFT..............................................................
void fftDestroy(void){
    cufftDestroy(fft2_r2c_zp);  cufftDestroy(fft2_c2r_zp); return;
}//END DESTROYER................................................................

// FORWARD FFT!.................................................................
void fftForward(double2* buffer){
    // Execute the transform
    cufftCheck(cufftExecD2Z(fft2_r2c_zp,(double*)buffer,(double2*)buffer),"forward transform_zp");
    // Normalize 
    normalize(buffer,(double)NTP*(2*NZP-2),NR*NTP*NZP);    return;
}// END FORWARD FFT.............................................................

// BACKWARD FFT!................................................................
void fftBackward(double2* buffer){
    // Execute the transform
    cufftCheck(cufftExecZ2D(fft2_c2r_zp,(double2*)buffer,(double*)buffer),"backward transform_zp");
    return;
}// END BACKWARD FFT............................................................

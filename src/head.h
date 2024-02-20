/* INTEGRATE NAVIER-STOKES EQUATIONS IN CYLINDRICAL COORDINATES USING A PSEUDO-SPECTRAL METHOD */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cufft.h>
//#include <hdf5.h>
//#include <hdf5_hl.h>
#include <time.h>
#include <sys/time.h>
/*++++++++++++++++++++++++++++++ CONSTANTS ++++++++++++++++++++++++++++*/
#define PI2 6.283185307179586   // two times pi

/*++++++++++++++++++++++++ SIMULATION PARAMETERS ++++++++++++++++++++++*/
// Grid points and modes
#define NR 48                 // Radial grid points
#define NT 64                 // 2*m_theta theta modes (must be even!)
#define NZ 257                // m_z+1     axial modes (must be odd! ) 

// Length of the pipe
#define LT PI2                 // Theta length
#define LZ 100.0               // Length of the pipe in R

// Physical parameters
#define Re  1850               // Reynolds number
#define A_P 0.3               // Initial perturbation amplitude

/*+++++++++++++++++++++++++++ IN&OUT PARAMETERS +++++++++++++++++++++++*/
// Restart 
#define restart 0            // >0 will try to read *_s.bin files

// Friction file
#define dt_frc 0.5             // t in code units to write to io_friction
#define dc_frc 100             // Number of rows to store before writing
// Q&U CrossS file
#define dt_qcr 0.5             // t in code units to write to io_q&ucross
#define dc_qcr 100             // Number of rows to store before writing
// Mean file (mean profile file)
#define dt_mnp 0.5             // t in code units to write to io_meanpr
#define dc_mnp 100             // Number of rows to store before writing

/*++++++++++++++++++++++++ INTEGRATOR PARAMETERS ++++++++++++++++++++++*/
// Finite differences
#define sten 7                  // Stencil of radial derivative
#define iw sten/2               // half stencil

// Time-Step size
#define dt    0.01              // time step size (constant)
#define d_im  0.51               // Factor multiplying the implicit term
#define nsteps 1000000             // Number of time steps
#define maxit  10               // Number of max iterations to converge
#define tol   5e-8               // Tolerance of convergence analysis

// Points after aliasing
#define NTP (3*NT/2)            // Total of physical azimuthal points
#define NZP (3*(NZ-1)/2+1)            // half of physical axial points

/*++++++++++++++++++++++ ARCHITECTURE PARAMETERS ++++++++++++++++++++++*/
// Prepare block dimensions
#define block_size  256         // Number of threads per block
#define grid_size 65535         // Number of blocks in direction y and z 

// Check Cudart in case u want to troubleshoot errors
#define CHECK_CUDART(x) do { \
  cudaError_t res = (x); \
  if(res != cudaSuccess) { \
    fprintf(stderr, " CUDART: %s = %d (%s) at (%s:%d)\n",  #x, res, cudaGetErrorString(res),__FILE__,__LINE__); \
    exit(1); \
  } \
} while(0)

/*+++++++++++++++    STRUCTURES INHERITED FROM ALBERTO   ++++++++++++++*/
typedef struct {double2* r;double2* t;double2* z;} vfield;
typedef struct {int Nr; int Nt; int Nz; double Lt ; double Lz;} size_p;

/*+++++++++++++++++++++++++++    FUNCTIONS    +++++++++++++++++++++++++*/
// BOUNDARY CONDITION
void setBCon(void);
void BConDestroy(void);
void corBCon(vfield u);

// DERIVATIVES
void setDeriv(double *r_h);
void derivt(double2 *grad,double2 *p);
void derivz(double2 *grad,double2 *p);
void DerivDestroy(void);
void initField(vfield u,vfield uw,vfield rhsw);

// FAST FOURIER TRANSFORM
void setFft(size_p sizes);
void fftDestroy(void);
void fftForward(double2* buffer);
void fftBackward(double2* buffer);

// CUBLAS ANDFLUX
void setCublasFlux(double *rdr_h);
void dotCub(double Qa,double* u);
void FluxDestroy(void);
void adj_flux(double2 *uz);
void cublasCheck(cublasStatus_t error, const char* function );
void iden_max(double *err,double *d1,double* d2);
void intq(double *qin,double *q,double *qaux);

// INOUT
void setIO(double *r_h);
void initField(vfield u);
void IODestroy(void);
void wrt_frc(double t,double2 *uz);
void writeH5file(vfield u,double time,const char* Nme);
void wBufferBinary(double* w,const char* file,size_t elsize,size_t elements);
void rBufferBinary(double* w,const char* file,size_t elsize,size_t elements);
void wrt_qcr(vfield u,double time);
void wrt_ucr(vfield u,double time);
void wrt_mpr(vfield u,double time);

// INTEGRATE
void setInt(void);
void IntDestroy(void);
void integrate(vfield u,vfield rhs);

// LINEAR
void setLinear(double *r_h,double *Lu_h,double *Lp_h);
void LinearDestroy(void);
void addLaplacian(vfield rhs,vfield u);
void LUsolve_u(double2 *sol,double2 *RHS,int flag);
void LUsolve_p(double2 *sol,double2 *RHS);

// NONLINEAR
void setNonlinear(void);
void NonDestroy(void);
void nonlinear(vfield u,vfield du);
void padForward(double2 *pad,double2 *u);
void padBackward(double2 *u,double2 *pad);

//RADIAL OPERATIONS
void init_fd(double *r_h,double *Lu_h,double *Lp_h,double *rdr_h);
void get_dr0(double *dr0_h,double *r_h);
void get_dr1(double *dr1);
void derivr(double2 *dur,double2 *u,int flag);
void divWall(double2 *div,vfield u);
void divergence(double2 *div,vfield u);
void addGrad(vfield rhs,double2 *p);
void pressGrad(double2 *duz,double2* uz);

// UTILS
void normalize(double2* u,double norm,size_t elements);
void uniformVec(double2 *u,double2 uni);
void write_bcon(double2 *rhs,double2 bcon);
void decoupleForward(vfield u);      //from r/theta to +/-
void decoupleBackward(vfield u);     //from +/- to r/theta
void copyBuffer(double2* u2, double2* u1); 
void copyVfield(vfield u2, vfield u1);

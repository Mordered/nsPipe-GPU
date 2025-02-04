/* Re-mesh the nsPipe code to a different grid */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
/*++++++++++++++++++++++++++++ CONSTANTS ++++++++++++++++++++++++++++*/
#define PI2 6.283185307179586   // two times pi

/*++++++++++++++++++++++++ OLD GRID PARAMETERS ++++++++++++++++++++++*/
// Grid points and modes
#define NR 48                 // Radial grid points
#define NT 64                 // 2*m_theta theta modes (must be even!)
#define NZ 257                // m_z+1     axial modes (must be odd! ) 

/*+++++++++++++++++++++++ DESIRED GRID PARAMETERS +++++++++++++++++++*/
#define Nrd 64                 // Radial grid points
#define Ntd 96                 // 2*m_theta theta modes (must be even!)
#define Nzd 301                // m_z+1     axial modes (must be odd! )

/*+++++++++++++++++++++ INTERPOLATOR PARAMETERS ++++++++++++++++++++++*/
// Half the total stencil to use for the interpolation
#define iw 3               // half stencil


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
void wBufferBinary(double* w,const char* file,size_t elsize,size_t elements);
void rBufferBinary(double* w,const char* file,size_t elsize,size_t elements);


//RE-MESH TOOL
void re_mesh(vfield u);

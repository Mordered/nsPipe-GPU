#include"head.h"
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
// PURPOSE:----------- SUBROUTINES to perform input/output 
// AUTHOR: ----------- Daniel Morón Montesdeoca
// DATE:   ----------- 24/04/2022
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
FILE *fp;
/*+++++++++++++++++++++++++++++ WRITE BUFFER +++++++++++++++++++++++++++++*/
// WRITE BUFFER............................................................/
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
}// END BUFFER WRITER....................................................../

/*+++++++++++++++++++++++++++++++++ READ BUFFER ++++++++++++++++++++++++++*/
// READ BUFFER............................................................./
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
}// END BUFFER READER....................................................../


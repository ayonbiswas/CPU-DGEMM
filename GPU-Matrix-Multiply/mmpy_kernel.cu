// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
#include "mytypes.h"
#include <stdio.h> 
#include <iostream> 

using namespace std;


__global__ void matMulnaive(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _DOUBLE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _DOUBLE_ a = A[I * N + k];
            _DOUBLE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}


template<int BN, int BM, int BK, int TN, int TM>
__device__ __inline__ void DGEMM(int N, _DOUBLE_ *__restrict__ C,const _DOUBLE_ * __restrict__ A,const _DOUBLE_ * __restrict__ B) {

    __shared__ _DOUBLE_ As[BM][BK];
    __shared__ _DOUBLE_ Bs[BK][BN];

    _DOUBLE_ a1, a2, a3,a4, b1, b2;  
    _DOUBLE_ tile_C[TM][TN] ={0};

    int x_blkA, x_blkB, y_blkA, y_blkB, x_blkC, y_blkC;
    int sh_xB, sh_yA;
    int btx, aty;
        
    //mapping thread blocks
    const int z = 0;
    x_blkA = z;
    y_blkA = BM*blockIdx.y;
    x_blkB = BN*blockIdx.x;
    y_blkB = z;


    btx = threadIdx.x;
    aty = threadIdx.y;
    x_blkC = x_blkB + TN*btx;
    y_blkC = y_blkA + TM*aty;

    //moving to shared mem
    #pragma unroll
    for(int i=z; i < N; i+=BK){       
    
        #pragma unroll
        for(int j=z;j<(BM+blockDim.y-1)/blockDim.y ;j++){        
            sh_yA =j*blockDim.y+ threadIdx.y;
            if(y_blkA+sh_yA < N && x_blkA + btx <N){
                *(*(As+sh_yA)+btx) = *(A +(y_blkA+sh_yA)*N +(x_blkA + btx));
            }
            else{
                *(*(As+sh_yA)+btx) = z;
            }
        }
        
        #pragma unroll
        for(int j=z; j<(BN+blockDim.x-1)/blockDim.x;j++){        
            sh_xB = j*blockDim.x+ threadIdx.x;
            if(x_blkB+sh_xB < N && y_blkB + aty <N){
                *(*(Bs+aty)+sh_xB) = *(B+ (y_blkB + aty)*N + (x_blkB+sh_xB));
            }
            else{
                *(*(Bs+aty)+sh_xB) = z;
            }
        }
        __syncthreads();
        x_blkA+=BK;
        y_blkB+=BK;

        //mapping threads; outerproduct        
        #pragma unroll
        for(int j =z;j<BK;j++){
            #pragma unroll
            for(int m =z;m<TM/4;m++){
                a1 = *(*(As+m+TM*aty)+j);
                a2 = *(*(As+m+TM/4+TM*aty)+j);
                a3 = *(*(As+m+TM/2+TM*aty)+j);
                a4 = *(*(As+m+(3*TM)/4+TM*aty)+j);

            #pragma unroll
                for(int n=z;n<TN/2;n++) { 
                    b1 = *(*(Bs+j)+n+TN*btx);
                    b2 = *(*(Bs+j)+n+TN/2+TN*btx);

                    *(*(tile_C+m)+n) += a1*b1;
                    *(*(tile_C+m)+n+TN/2) += a1*b2; 
                    *(*(tile_C+m+TM/4)+n) += a2*b1; 
                    *(*(tile_C+m+TM/4)+n+TN/2) += a2*b2;    
                    *(*(tile_C+m+TM/2)+n) += a3*b1;
                    *(*(tile_C+m+TM/2)+n+TN/2) += a3*b2; 
                    *(*(tile_C+m+(3*TM)/4)+n) += a4*b1; 
                    *(*(tile_C+m+(3*TM)/4)+n+TN/2) += a4*b2;             
                }
            }    
        }
        __syncthreads();
        
    }
    #pragma unroll
    for(int m =z;m<TM;m++){
        #pragma unroll
        for(int n=z;n<TN;n++){
            if(x_blkC+n<N && y_blkC+m<N){
                *(C+(x_blkC+n)+(y_blkC+m)*N) = *(*(tile_C + m)+n);
            }
        }
    }
  
}

__global__ void matMul(int N, _DOUBLE_ *__restrict__ C,const _DOUBLE_ * __restrict__ A,const _DOUBLE_ * __restrict__ B) {
    if(N<480){
        DGEMM<SBN,SBM, SBK, STN, STM>(N, C, A, B);
    }
    else{
        DGEMM<LBN,LBM, LBK, LTN, LTM>(N, C, A, B);
    }

}

template<int BN, int BM, int BK, int TN, int TM>
__global__ void matMulcutlass(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

    __shared__ _DOUBLE_ As[BM][BK];
    __shared__ _DOUBLE_ Bs[BK][BN];

    int x_blkA, x_blkB, y_blkA, y_blkB, x_blkC, y_blkC;
    int sh_xA, sh_xB, sh_yA, sh_yB;
    int btx, aty;

    _DOUBLE_ tile_C[2*TM][2*TN]={0};
    //
    x_blkA = 0;
    y_blkA = BM*blockIdx.y;
    x_blkB = BN*blockIdx.x;
    y_blkB = 0;

    //mapping thread blocks
    btx = 2*threadIdx.x;
    aty = 2*threadIdx.y;
    x_blkC = x_blkB + TN*btx;
    y_blkC = y_blkA + TM*aty;


    sh_xA = btx%BK;
    sh_yB = aty%BK;

    #pragma unroll
    for(int i=0; i < N; i+=BK){        
    
        #pragma unroll
        for(int j=0;j<(BM+blockDim.y-1)/blockDim.y ;j++){        
            sh_yA =j*blockDim.y+ threadIdx.y;
            if(y_blkA+sh_yA < N ){
                As[sh_yA][sh_xA] = A[ (y_blkA+sh_yA)*N +(x_blkA + sh_xA)];
                As[sh_yA][sh_xA+1] = A[ (y_blkA+sh_yA)*N +(x_blkA + sh_xA+1)];
            }
            else{
                As[sh_yA][sh_xA] = 0;
                As[sh_yA][sh_xA+1] = 0;
            }
   
        }
        
        #pragma unroll
        for(int j=0; j<(BN+blockDim.x-1)/blockDim.x;j++){        
            sh_xB = j*blockDim.x+ threadIdx.x;
            if(x_blkB+sh_xB < N){
                Bs[sh_yB][sh_xB] = B[ (y_blkB + sh_yB)*N + (x_blkB+sh_xB)];
                Bs[sh_yB+1][sh_xB] = B[ (y_blkB + sh_yB+1)*N + (x_blkB+sh_xB)];
            }
            else{
                Bs[sh_yB][sh_xB] = 0;
                Bs[sh_yB+1][sh_xB] = 0;

      
            }
        }
        __syncthreads();
        x_blkA+=BK;
        y_blkB+=BK;

        //mapping threads; outerproduct
        
        #pragma unroll
        for(int j =0;j<BK;j++){
            #pragma unroll
            for(int m =0;m<TM;m++){
                for(int n=0;n<TN;n++) { 
                    
                    tile_C[m][n] += As[m+TM*aty][j]*Bs[j][n+TN*btx];
                    tile_C[m][n+TN] += As[m+TM*aty][j]*Bs[j][n+TN*btx+TN];
                    tile_C[m+TM][n] += As[m+TM*aty+TM][j]*Bs[j][n+TN*btx];
                    tile_C[m+TM][n+TN] += As[m+TM*aty+TM][j]*Bs[j][n+TN*btx+TN];                
            
                }
            }    
        }
        __syncthreads();
        
    }
    #pragma unroll
    for(int m =0;m<TM;m++){
        #pragma unroll
        for(int n=0;n<TN;n++){
            if(x_blkC+n<N && y_blkC+m<N){
                C[(x_blkC+n)+(y_blkC+m)*N] = tile_C[m][n];
                C[(x_blkC+TN+n)+(y_blkC+m)*N] = tile_C[m][n+TN];
                C[(x_blkC+n)+(y_blkC+TM+m)*N] = tile_C[m+TM][n];
                C[(x_blkC+TN+n)+(y_blkC+TM+m)*N] = tile_C[m+TM][n+TN];
            }
        }
    }
}


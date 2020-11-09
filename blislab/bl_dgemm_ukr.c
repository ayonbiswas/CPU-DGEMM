#include "bl_config.h"
#include "bl_dgemm_kernel.h"
#include <immintrin.h>
#include <avx2intrin.h>

#define a(i, j, ld) a[ (j)*(ld) + (i) ]
#define b(i, j, ld) b[ (i)*(ld) + (j) ]
#define c(i, j, ld) c[ (i)*(ld) + (j) ]

//
// C-based micorkernel
//
void bl_dgemm_ukr( int    k,
                    int    m,
                   int    n,
                   const double * restrict a,
                   const double * restrict b,
                   double *c,
                   unsigned long long ldc,
                   aux_t* data )
{
    int l, j, i;

    for ( l = 0; l < k; ++l )
    {                 
        for ( j = 0; j < n; ++j )
        { 
            for ( i = 0; i < m; ++i )   
            { 
	        c( i, j, ldc ) += a( i, l, DGEMM_MR) * b( l, j, DGEMM_NR );   
            }
        }
    }

}

//final 4x8 ukr
inline void simd_4x8_ukr(int    kc,
                    int    m,
                   int    n,
                   const double * restrict a,
                   const double * restrict b,
                   double * c,
                   unsigned long long ldc,
                   aux_t* data
                   )
{   const int num_NR = DGEMM_NR/4;
    __m256d a_reg[DGEMM_MR],b_reg[num_NR],c_reg[DGEMM_MR*(num_NR)];

    for (int i=0;i<DGEMM_MR;i+=4){
        for (int j = 0;j<num_NR;j+=2){
            c_reg[(i+0)*num_NR+(j+0)] = _mm256_load_pd(c + (i+0)*ldc+4*(j+0));
            c_reg[(i+1)*num_NR+(j+0)] = _mm256_load_pd(c + (i+1)*ldc+4*(j+0));
            c_reg[(i+2)*num_NR+(j+0)] = _mm256_load_pd(c + (i+2)*ldc+4*(j+0));
            c_reg[(i+3)*num_NR+(j+0)] = _mm256_load_pd(c + (i+3)*ldc+4*(j+0));
            
            c_reg[(i+0)*num_NR+(j+1)] = _mm256_load_pd(c + (i+0)*ldc+4*(j+1));
            c_reg[(i+1)*num_NR+(j+1)] = _mm256_load_pd(c + (i+1)*ldc+4*(j+1));
            c_reg[(i+2)*num_NR+(j+1)] = _mm256_load_pd(c + (i+2)*ldc+4*(j+1));
            c_reg[(i+3)*num_NR+(j+1)] = _mm256_load_pd(c + (i+3)*ldc+4*(j+1));
        }
    }
    for(int ik=0;ik<kc;ik++){
        for(int j = 0; j<DGEMM_MR; j+=4){
            a_reg[j+0] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+0);
            a_reg[j+1] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+1);
            a_reg[j+2] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+2);
            a_reg[j+3] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+3);
        }
        for (int j = 0;j<num_NR;j+=2){
            b_reg[j+0] = _mm256_load_pd(b + (ik)*DGEMM_NR +4*(j+0) );
            b_reg[j+1] = _mm256_load_pd(b + (ik)*DGEMM_NR +4*(j+1) );
        }
        for(int it = 0; it<DGEMM_MR; it+=4){
            for (int j = 0;j<num_NR;j+=2){
                c_reg[(it+0)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+0],b_reg[j+0],c_reg[(it+0)*num_NR+(j+0)]);
                c_reg[(it+1)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+1],b_reg[j+0],c_reg[(it+1)*num_NR+(j+0)]);
                c_reg[(it+2)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+2],b_reg[j+0],c_reg[(it+2)*num_NR+(j+0)]);
                c_reg[(it+3)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+3],b_reg[j+0],c_reg[(it+3)*num_NR+(j+0)]);

                c_reg[(it+0)*num_NR+(j+1)] =  _mm256_fmadd_pd (a_reg[it+0],b_reg[j+1],c_reg[(it+0)*num_NR+(j+1)]);
                c_reg[(it+1)*num_NR+(j+1)] =  _mm256_fmadd_pd (a_reg[it+1],b_reg[j+1],c_reg[(it+1)*num_NR+(j+1)]);
                c_reg[(it+2)*num_NR+(j+1)] =  _mm256_fmadd_pd (a_reg[it+2],b_reg[j+1],c_reg[(it+2)*num_NR+(j+1)]);
                c_reg[(it+3)*num_NR+(j+1)] =  _mm256_fmadd_pd (a_reg[it+3],b_reg[j+1],c_reg[(it+3)*num_NR+(j+1)]);
            }
        }  

    }

    for(int ir = 0; ir<DGEMM_MR; ir+=4){
        for (int j = 0;j<num_NR;j+=2){
            _mm256_store_pd(c+ (ir+0)*ldc + 4*(j+0), c_reg[(ir+0)*num_NR+(j+0)]);
            _mm256_store_pd(c+ (ir+1)*ldc + 4*(j+0), c_reg[(ir+1)*num_NR+(j+0)]);
            _mm256_store_pd(c+ (ir+2)*ldc + 4*(j+0), c_reg[(ir+2)*num_NR+(j+0)]);
            _mm256_store_pd(c+ (ir+3)*ldc + 4*(j+0), c_reg[(ir+3)*num_NR+(j+0)]);

            _mm256_store_pd(c+ (ir+0)*ldc + 4*(j+1), c_reg[(ir+0)*num_NR+(j+1)]);
            _mm256_store_pd(c+ (ir+1)*ldc + 4*(j+1), c_reg[(ir+1)*num_NR+(j+1)]);
            _mm256_store_pd(c+ (ir+2)*ldc + 4*(j+1), c_reg[(ir+2)*num_NR+(j+1)]);
            _mm256_store_pd(c+ (ir+3)*ldc + 4*(j+1), c_reg[(ir+3)*num_NR+(j+1)]);
            
        }
    }

}

//for testing 4x4 and 4x16 ukr
inline void simd_4x4m_ukr(int    kc,
                    int    m,
                   int    n,
                   const double * restrict a,
                   const double * restrict b,
                   double * c,
                   unsigned long long ldc,
                   aux_t* data
                   )
{   const int num_NR = DGEMM_NR/4;
    __m256d a_reg[DGEMM_MR],b_reg[num_NR],c_reg[DGEMM_MR*(num_NR)];


    for (int i=0;i<DGEMM_MR;i+=4){
        for (int j = 0;j<num_NR;j++){
            c_reg[(i+0)*num_NR+(j+0)] = _mm256_load_pd(c + (i+0)*ldc+4*(j+0));
            c_reg[(i+1)*num_NR+(j+0)] = _mm256_load_pd(c + (i+1)*ldc+4*(j+0));
            c_reg[(i+2)*num_NR+(j+0)] = _mm256_load_pd(c + (i+2)*ldc+4*(j+0));
            c_reg[(i+3)*num_NR+(j+0)] = _mm256_load_pd(c + (i+3)*ldc+4*(j+0));            
        }
    }
    for(int ik=0;ik<kc;ik++){
        for(int j = 0; j<DGEMM_MR; j+=4){
            a_reg[j+0] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+0);
            a_reg[j+1] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+1);
            a_reg[j+2] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+2);
            a_reg[j+3] =  _mm256_broadcast_sd ( a + ik*DGEMM_MR +j+3);
        }

        for (int j = 0;j<num_NR;j++){
            b_reg[j+0] = _mm256_load_pd(b + (ik)*DGEMM_NR +4*(j+0) );
        }

        for(int it = 0; it<DGEMM_MR; it+=4){
            for (int j = 0;j<num_NR;j++){
                c_reg[(it+0)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+0],b_reg[j+0],c_reg[(it+0)*num_NR+(j+0)]);
                c_reg[(it+1)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+1],b_reg[j+0],c_reg[(it+1)*num_NR+(j+0)]);
                c_reg[(it+2)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+2],b_reg[j+0],c_reg[(it+2)*num_NR+(j+0)]);
                c_reg[(it+3)*num_NR+(j+0)] =  _mm256_fmadd_pd (a_reg[it+3],b_reg[j+0],c_reg[(it+3)*num_NR+(j+0)]);
            }
        }  

    }

    for(int ir = 0; ir<DGEMM_MR; ir+=4){
        for (int j = 0;j<num_NR;j++){
            _mm256_store_pd(c+ (ir+0)*ldc + 4*(j+0), c_reg[(ir+0)*num_NR+(j+0)]);
            _mm256_store_pd(c+ (ir+1)*ldc + 4*(j+0), c_reg[(ir+1)*num_NR+(j+0)]);
            _mm256_store_pd(c+ (ir+2)*ldc + 4*(j+0), c_reg[(ir+2)*num_NR+(j+0)]);
            _mm256_store_pd(c+ (ir+3)*ldc + 4*(j+0), c_reg[(ir+3)*num_NR+(j+0)]); 
        }
    }

}

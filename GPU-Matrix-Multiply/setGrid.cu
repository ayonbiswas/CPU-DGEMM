
#include "mytypes.h"
#include <stdio.h> 
#include <iostream> 
using namespace std;


void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{  
   int BN, BM, BK, TN, TM;
   if(n < 320){
      BN = SBN; BM = SBM; BK = SBK; TN = STN; TM = STM;
   }
   else{
      BN = LBN; BM = LBM; BK = LBK; TN = LTN; TM = LTM;
   }
   blockDim.x = BN/(TN);
   blockDim.y = BM/(TM);   
   gridDim.x = n/BN;
   gridDim.y = n/BM;
   if(n % blockDim.x != 0)
   	gridDim.x++;
   if(n % blockDim.y != 0)
    	gridDim.y++;

}

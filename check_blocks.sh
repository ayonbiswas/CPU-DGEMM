#!/bin/bash
#Generate data.txt
#May need to set execute permission: "chmod +x genDATA.sh"
set -e
matrix_sizes=(32 64 128 256 300 512 513 700 1023 1024 1300 2048)

block_length_l1=(32 64 128 256 512 1024)
block_length_l2=(8 16 32 64 128 256)
block_length_l3=(1024 2048 4096 8192 15360 30720)

make clean
for mc in "${block_length_l3[@]}"
do
	> "mc_data_$mc.txt"
	make MY_OPT="-O3 -DDGEMM_NC=64 -DDGEMM_KC=256 -DDGEMM_MC=$mc -DDGEMM_MR=4 -DDGEMM_NR=8" benchmark-blislab
	for size in "${matrix_sizes[@]}"
	do	

		SUM=0
		for i in {1..20}
		do
			
			OUTPUT=$(./benchmark-blislab -n $size -g)
			GFLOPS=$(echo $OUTPUT | awk '{print $2}')
			SUM=$(awk '{print $1+$2}' <<<"${SUM} ${GFLOPS}")
		done
		AVG=$(awk '{print $1/20}' <<<"${SUM}")
		printf '%d\t%s\n' $size $AVG >> "mc_data_$mc.txt"
		
    done
	make clean
done
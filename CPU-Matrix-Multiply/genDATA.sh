#!/bin/bash
#Generate data.txt
#May need to set execute permission: "chmod +x genDATA.sh"
matrix_sizes=(32 33 63 64 128 130 256 300 511 512 513 700 900 1023 1024 1025 1300 1700 2047 2048)

> data4x16.txt

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
	printf '%d\t%s\n' $size $AVG >> data4x16.txt
done


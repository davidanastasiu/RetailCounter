#!/bin/sh

for i in $(seq -f "%05g" 1 116)
do
	filename="$i*_*"
	mkdir $i
	mv $filename "$i/"
done

#!/bin/bash

#for each .h5 file in the path, run the following analysis commands
for i in `\ls -R /project/bi_dsc_community/calorimeter/LEMURS/all_data/*/*/*.h5`; do 
	#extract file name from the path
	filename=`echo $i | cut -d '/' -f 9 | cut -d '_' -f 2- | cut -d '.' -f 1-3 ;`

	python lemurs_analysis.py corr --data $i  --n-energy-bins 10 --n-theta-bins 10 --n-phi-bins 10 --min-samples 50 --save $filename'_stratified_correlations.pdf'
	python lemurs_analysis.py means --data $i --n-energy-bins 10 --n-theta-bins 10 --n-phi-bins 10   --min-samples 50 --normalize-per-event   --save $filename'_stratified_mean_profiles.pdf'
	python lemurs_analysis.py phi-r --data $i --n-theta-bins 10 --n-phi-bins 4 --min-samples 50  --save-phi $filename+'_angular_bin_shifts_analysis.pdf'  --save-r  $filename'_radial_bin_shifts_analysis.pdf'
	python lemurs_analysis.py layer-shifts --data $i --n-theta-bins 10 --n-phi-bins 4 --min-samples 50   --save $filename'_layer_shifts_analysis.pdf'
	python lemurs_analysis.py global-corr --data $i --save $filename'_global_correlations.pdf'

done

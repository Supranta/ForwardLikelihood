#!/bin/bash
set -e
echo "$@"

filename="$1"
shift
echo "$@"
name=$(echo "$filename" | rev | cut -d'/' -f1 | rev) # get the stem
# sigint=$(grep 'sigint' "$filename" | cut -d= -f2 | cut -d' ' -f2) # extract sigma_int from fitres
echo -e "\nProcessing $name..."
# echo "sigint: $sigint"
python ../halo_sim/make_datafile.py "$filename" -o tmp.csv "$@" # make datafile
sed -e "s/<file>/tmp.csv/g" real_template.ini > tmp.ini # make temporary input file
python fit.py tmp.ini # run fit
mv ./output/fitres/results.txt ./output/fitres/"$name".txt # put results in sensible place
rm tmp.csv tmp.ini

#!/bin/bash
set -e
echo "$@"

directory="$1"
shift
echo "$@"

rm -f output/fitres/*

for filename in "$directory"/*.FITRES
do
    name=$(echo "$filename" | rev | cut -d'/' -f1 | rev) # get the stem
    sigint=$(grep 'sigint' "$filename" | cut -d= -f2 | cut -d' ' -f2) # extract sigma_int from fitres
    echo -e "\nProcessing $name..."
    echo "sigint: $sigint"
    python ../halo_sim/make_datafile.py "$filename" -o tmp.csv "$@" --convert  #--flux 10000 # make datafile
    sed -e "s/<file>/tmp.csv/g;s/<sigint>/$sigint/g" template.ini > tmp.ini # make temporary input file
    python fit.py tmp.ini # run fit
    mv ./output/fitres/results.txt ./output/fitres/"$name".txt # put results in sensible place
    rm tmp.csv tmp.ini
done

sum=0
sum2=0
i=0

for result in ./output/fitres/*.txt
do
    beta=$(grep beta "$result" | sed -re "s/beta: (-?([0-9]*[.])?[0-9]+).*/\1/")
    echo "$beta"
    sum=$(python -c "print($sum + $beta)")
    sum2=$(python -c "print($sum2 + $beta**2)")
    ((i=i+1))
done

avg=$(python -c "print($sum/$i)")
std=$(python -c "import math; print(math.sqrt($sum2/$i - $avg**2))")

echo "Mean: $avg"
echo "Std. dev.: $std"

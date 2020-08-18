#!/bin/bash
set -e

python raw_test/raw_test.py -n 300 -s 100 --seed 12345

for filename in raw_test/out/*.csv
do
    name=$(echo "$filename" | rev | cut -d'/' -f1 | rev) # get the stem
    echo -e "\nProcessing $filename..."
    sed -e "s|<file>|$filename|g" raw_test.ini > tmp.ini
    python fit.py tmp.ini
    mv ./output/raw_test/results.txt ./output/raw_test/"$name".txt
    rm tmp.ini
done

sum=0
sum2=0
i=0

for result in ./output/raw_test/*.txt
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

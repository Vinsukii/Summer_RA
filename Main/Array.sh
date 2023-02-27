runs=30
method='NE_2.1'
algo='erl2.1'
sizes=( '1005' '2005' '1510' '2010' '3010' '4010' 'Public' )

for size in "${sizes[@]}"
do
    echo SIZE: $size
    for algo in "${algos[@]}"
    do
        echo ALGO: $algo
        qsub -t 1-$runs:1 Job_NN.sh $method $size $algo
    done
done

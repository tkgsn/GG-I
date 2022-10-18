locations=(Akita Tokyo)
distances=(1000 2000 3000 4000)
mechanisms=(OPT_GEOI OPT_GEM GEM)
epsilons=(0.001 0.005 0.01 0.05 0.1)

for location in ${locations[@]}
do
    for distance in ${distances[@]}
    do
        for mechanism in ${mechanisms[@]}
        do
            for epsilon in ${epsilons[@]}
            do
                if [ $location = Tokyo ]; then
                    python run.py --mechanism $mechanism --location $location --simplify --distance $distance --epsilon $epsilon
                else
                    python run.py --mechanism $mechanism --location $location --distance $distance --epsilon $epsilon
                fi
            done
        done
    done
done
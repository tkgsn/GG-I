locations=(Akita Tokyo)
distances=(10000)
mechanisms=(OPT_GEOI OPT_GEOGI OPT_GEM GEM)
epsilons=(0.0001 0.0005 0.001 0.005 0.01)
delta=1.1
n_graph_nodes=1000
n_optgeoi_nodes=100

for location in ${locations[@]}
do
    for distance in ${distances[@]}
    do
        for mechanism in ${mechanisms[@]}
        do
            for epsilon in ${epsilons[@]}
            do
                if [ $location = Tokyo ]; then
                    python run.py --mechanism $mechanism --location $location --simplify --distance $distance --epsilon $epsilon --delta $delta --n_graph_nodes $n_graph_nodes --n_optgeoi_nodes $n_optgeoi_nodes
                else
                    python run.py --mechanism $mechanism --location $location --distance $distance --epsilon $epsilon --delta $delta --n_graph_nodes $n_graph_nodes --n_optgeoi_nodes $n_optgeoi_nodes
                fi
            done
        done
    done
done
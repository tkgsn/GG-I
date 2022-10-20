locations=(Akita Tokyo)
distances=(10000)
mechanisms=(OPT_GEOI OPT_GEOGI OPT_GEM GEM)
epsilons=(0.0001 0.0005 0.001 0.005 0.01)
delta=1.1
n_graph_nodes=1000
n_optgeoi_nodess=(10 50 100 150 200)


for epsilon in ${epsilons[@]}
do
    for distance in ${distances[@]}
    do
        for location in ${locations[@]}
        do
            for mechanism in ${mechanisms[@]}
            do
                if [ $mechanism = GEM ] || [ $mechanism = OPT_GEM ]; then
                    if [ $location = Tokyo ]; then
                        python run.py --mechanism $mechanism --location $location --simplify --distance $distance --epsilon $epsilon --delta $delta --n_graph_nodes $n_graph_nodes
                    else
                        python run.py --mechanism $mechanism --location $location --distance $distance --epsilon $epsilon --delta $delta --n_graph_nodes $n_graph_nodes
                    fi
                else
                    for n_optgeoi_nodes in ${n_optgeoi_nodess[@]}
                    do
                        if [ $location = Tokyo ]; then
                            python run.py --mechanism $mechanism --location $location --simplify --distance $distance --epsilon $epsilon --delta $delta --n_graph_nodes $n_graph_nodes --n_optgeoi_nodes $n_optgeoi_nodes
                        else
                            python run.py --mechanism $mechanism --location $location --distance $distance --epsilon $epsilon --delta $delta --n_graph_nodes $n_graph_nodes --n_optgeoi_nodes $n_optgeoi_nodes
                        fi
                    done
                fi
                
            done
        done
    done
done
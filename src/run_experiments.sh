echo "===== MNIST runs ====="

for i in {0..9}
do
    echo "MNIST seed $i"
    python3 train.py --config config_mnist.yaml --seed $i > logs_sh/logs_mnist_seed_$i.txt
done


echo "===== OCTMNIST runs ====="

for i in {0..9}
do
    echo "OCTMNIST seed $i"
    python3 train.py --config config_octmnist.yaml --seed $i > logs_sh/logs_octmnist_seed_$i.txt
done


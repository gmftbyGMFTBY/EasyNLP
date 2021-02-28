mode=$1    # init / retrieval
python process.py --seed 50 --name lccc --train_size 500000 --test_size 1000 --database_size 1000000 --mode $mode --samples 10

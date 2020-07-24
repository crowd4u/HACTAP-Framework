# HACTAP-Framework
Algorithms for Human+AI Crowd Task Assignment Problem

## setup
```
pip install -e .
```

## test
```
python -m unittest
```

or

```
python -m unittest tests.test_utils
```


## memo
```
nohup parallel --joblog ./jobs.log --result ./parallel_out -j 2 -a batch_experiment_mnist.txt &
```
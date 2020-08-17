# Experiment with MIND dataset

## solo
```
python experiment.py --task_size 10240 --human_crowd_batch_size 1000 --quality_requirements 0.9 --solver gta
```

## download dataset
```
todo: add hogehoge.py
```

## batch

```
parallel --joblog ./jobs.log --result ./parallel_out -j 1 'python experiment.py --group_id demo --trial_id {1} --task_size 10240 --human_crowd_batch_size 1000 --quality_requirements {2} --solver {3}' ::: {1..5} ::: 0.85 0.9 0.95 ::: gta al
```

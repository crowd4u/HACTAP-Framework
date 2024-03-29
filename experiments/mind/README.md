# Experiment with MIND dataset

## setup
for gpu
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## download dataset
```
# download original images
aws s3 cp s3://projects.crowd4u.org/mind/flood_image/ ./original_images --recursive --profile=mlab

# create dataset directories
python prepare_dataset.py
```

## solo
```
python experiment.py --human_crowd_batch_size 200 --quality_requirements 0.95 --solver gta --dataset mind-10-amt --human_crowd_mode random
```

## batch

Batch experiment using mind-10 dataset
```
parallel --joblog ./jobs.log --result ./parallel_out -j 1 'python experiment.py --group_id comparison_experiment_mind --trial_id {1} --human_crowd_batch_size 500 --quality_requirements {2} --solver {3} --human_crowd_mode {4} --dataset {5}' ::: {1..5} ::: 0.8 0.85 0.9 0.95 ::: gta ala ::: order random ::: mind-10 mind-10-amt
```


parallel --joblog ./jobs.log --result ./parallel_out_comp1 -j 1 'python experiment.py --group_id comparison_experiment_mind --trial_id {1} --human_crowd_batch_size 500 --quality_requirements {2} --solver {3} --human_crowd_mode {4} --dataset {5}' ::: {1..5} ::: 0.8 0.85 0.9 0.95 ::: gta ala ::: order ::: mind-10


Batch experiment using 106 dataset
```
parallel --joblog ./jobs.log --result ./parallel_out -j 1 'python experiment.py --group_id test_106_dataset --trial_id {1} --human_crowd_batch_size 40000 --quality_requirements {2} --solver {3} --human_crowd_mode {4} --dataset {5}' ::: 5 ::: 0.8 0.85 0.9 0.95 ::: gta al ::: order random ::: mind-106 mind-106-amt
```

Show log files
```
find parallel_out -name 'stdout' | xargs tail
```

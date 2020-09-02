# Experiment with MIND dataset

## download dataset
```
# download original images
aws s3 cp s3://projects.crowd4u.org/mind/flood_image/ ./original_images --recursive --profile=mlab
python prepare_dataset.py
```

## solo
```
python experiment.py --human_crowd_batch_size 500 --quality_requirements 0.95 --solver al --dataset mind-10-amt --human_crowd_mode random
```

## batch

```
parallel --joblog ./jobs.log --result ./parallel_out -j 1 'python experiment.py --group_id test_dataset --trial_id {1} --human_crowd_batch_size 40000 --quality_requirements {2} --solver {3} --human_crowd_mode {4} --dataset {5}' ::: 1 ::: 0.85 0.9 0.95 ::: gta al ::: order ::: mind-10 mind-10-amt
```

Batch experiment using 10 dataset
```
parallel --joblog ./jobs2.log --result ./parallel_out2 -j 1 'python experiment.py --group_id test_10_dataset --trial_id {1} --human_crowd_batch_size 500 --quality_requirements {2} --solver {3} --human_crowd_mode {4} --dataset {5}' ::: {1..10} ::: 0.8 0.85 0.9 0.95 ::: gta al ::: order random ::: mind-10 mind-10-amt
```

Batch experiment using 106 dataset
```
parallel --joblog ./jobs.log --result ./parallel_out -j 1 'python experiment.py --group_id test_106_dataset --trial_id {1} --human_crowd_batch_size 40000 --quality_requirements {2} --solver {3} --human_crowd_mode {4} --dataset {5}' ::: 5 ::: 0.8 0.85 0.9 0.95 ::: gta al ::: order random ::: mind-106 mind-106-amt
```

Show log files
```
find parallel_out -name 'stdout' | xargs tail
```

## memo
```
confusion_matrix 
[[31382    42   159]
 [  489   147     6]
 [  132     0   939]]
classification_report               precision    recall  f1-score   support

           0       0.98      0.99      0.99     31583
           1       0.78      0.23      0.35       642
           2       0.85      0.88      0.86      1071

    accuracy                           0.98     33296
   macro avg       0.87      0.70      0.73     33296
weighted avg       0.97      0.98      0.97     33296
```

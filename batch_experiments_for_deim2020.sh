#!/bin/bash

set -eu

trap 'curl -X POST --data-urlencode "payload={\"channel\": \"#experiments\", \"username\": \"HACTAP BOT\", \"text\": \":scream: Some error occurred.\", \"icon_emoji\": \":satellite:\"}" https://hooks.slack.com/services/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' ERR

group_id="deim2020"

curl -X POST --data-urlencode "payload={\"channel\": \"#experiments\", \"username\": \"HACTAP BOT\", \"text\": \":rocket: Start batch script (group_id: ${group_id})\", \"icon_emoji\": \":satellite:\"}" https://hooks.slack.com/services/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

for trial_id in `seq 0 4`
do

python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.7 --human_query_strategy random --enable_task_cluster 1
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.8 --human_query_strategy random --enable_task_cluster 1
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.9 --human_query_strategy random --enable_task_cluster 1

python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.7 --human_query_strategy random --enable_task_cluster 1 --enable_quality_guarantee 1
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.8 --human_query_strategy random --enable_task_cluster 1 --enable_quality_guarantee 1
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.9 --human_query_strategy random --enable_task_cluster 1 --enable_quality_guarantee 1

python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.7 --human_query_strategy random --enable_task_cluster 0
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.8 --human_query_strategy random --enable_task_cluster 0
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.9 --human_query_strategy random --enable_task_cluster 0

python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.7 --human_query_strategy uncertainty_sampling --enable_task_cluster 0
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.8 --human_query_strategy uncertainty_sampling --enable_task_cluster 0
python experiment.py --group_id ${group_id} --trial_id ${trial_id} --task_size 10000 --human_crowd_batch_size 200 --quality_requirements 0.9 --human_query_strategy uncertainty_sampling --enable_task_cluster 0

done

curl -X POST --data-urlencode "payload={\"channel\": \"#experiments\", \"username\": \"HACTAP BOT\", \"text\": \":checkered_flag: Finished batch script (group_id: ${group_id})\", \"icon_emoji\": \":satellite:\"}" https://hooks.slack.com/services/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

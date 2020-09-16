import json
import urllib.request
from PIL import Image
import glob
import os
import shutil
import pandas as pd
from dateutil import parser
import time
from itertools import groupby
from collections import Counter

target_for_10 = [
    '0075-qv',
    '0072-qv',
    '0052-qv',
    '0145-qv',
    '0083-qv',
    '0091-qv',
    '0061-qv',
    '0074-qv',
    '0130-qv',
    '0047-qv'
]


def sort_function(a):
    return a['image_id']


def most_frequent(target):
    occurence_count = Counter(target)
    return occurence_count.most_common(1)[0][0]


def split_image(image_i, image_and_status_ids, result_table, data_dir):
    height = 122
    width = 110

    # print(image_i)
    target_image = Image.open(image_i)
    file_id = image_i.split('/')[1][0:4]
    # print(file_id)

    for h_i in range(32):
        for w_i in range(32):
            w_k = w_i * width
            h_k = h_i * height

            cropped = target_image.crop((
                w_k,
                h_k,
                width + w_k,
                height + h_k
            ))

            if file_id not in image_and_status_ids or image_and_status_ids[file_id][h_i][w_i] == None: # NOQA
                label = 3
                timestamp = -1
            else:
                # print(image_and_status_ids[file_id][h_i][w_i])
                label = image_and_status_ids[file_id][h_i][w_i][0]
                timestamp = image_and_status_ids[file_id][h_i][w_i][1]

            if label == 1:
                label = 0

            if label == 2:
                label = 1

            if label == 4:
                label = 2

            output_name = './dataset/{}/{}/{}-qv_6_{}_{}.png'.format(
                data_dir, label, file_id, h_i + 1, w_i + 1
            )
            print(output_name)
            cropped.save(output_name)

            if label != 3:
                row = pd.Series(
                    [
                        output_name,
                        timestamp,
                    ],
                    index=['path', 'created_at']
                )

                result_table = result_table.append(row, ignore_index=True)

    return result_table


def get_data_from_crowd4u(project_name, relation_name):
    root_url = "http://crowd4u.org/api/relation_data"
    params = {
        "project_name": project_name,
        "relation_name": relation_name,
    }

    request = urllib.request.Request(
        '{}?{}'.format(root_url, urllib.parse.urlencode(params))
    )

    with urllib.request.urlopen(request) as res:
        datajson = json.load(res)

    return datajson["data"]


def mind_106():
    project_name = 'cyber_disaster_drill2'

    images = get_data_from_crowd4u(project_name, 'Image')
    answers = get_data_from_crowd4u(project_name, 'Answer')
    print('images', len(images), images[0])
    print('answers', len(answers), answers[0])

    list_image_and_label = []

    sorted_answers = sorted(answers, key=sort_function)
    for key, image_i in groupby(sorted_answers, key=sort_function):
        status_ids = []
        list_timestamp = []

        for image_i_answer_j in image_i:
            timestamp = image_i_answer_j['answered_at']

            if not timestamp:
                timestamp = image_i_answer_j['updated_at']

            # print(timestamp)
            if timestamp.startswith('2019-10-07') or timestamp.startswith('2019-10-08'): # NOQA
                # print(image_i_answer_j)
                status_ids.append(image_i_answer_j['status_id'])
                list_timestamp.append(timestamp)

        if len(status_ids) > 6:
            # print(key, status_ids, list_timestamp)
            the_status_id = most_frequent(status_ids)
            the_timestamp = int(time.mktime(parser.parse(
                list_timestamp[0]
            ).timetuple()))

            timelapse_start = 1570482000
            timelapse_end = 1570525200

            in_time = (timelapse_start < the_timestamp and timelapse_end > the_timestamp ) # NOQA

            if the_status_id != 3 and in_time:
                the_image = next(filter(
                    lambda x: x['image_id'] == key, images
                ))
                # print(
                # key, the_status_id, list_timestamp[0], the_image['image_url']
                # )

                list_image_and_label.append((
                    the_image['image_url'],
                    the_status_id,
                    the_timestamp
                ))

    # print(list_image_and_label)
    image_and_label = {}

    for image_url, status, timestamp in list_image_and_label:
        image_name = image_url.replace(
            "projects.crowd4u.org/mind/img/{}/dest/".format(project_name), ''
        ).replace('.png', '')

        image_index = image_name[0:4]

        nest_level, loc_y, loc_x = image_name.split('-qv_')[1].split('_')
        nest_level = int(nest_level)
        loc_y = int(loc_y)
        loc_x = int(loc_x)
        loc_padding = int(32 / (2 ** (nest_level - 1)))
        # print(nest_level, loc_y, loc_x)

        if image_index not in image_and_label:
            image_and_label[image_index] = [
                [None for x in range(32)] for y in range(32)
            ]

        if status in [1, 2, 4]:
            for y_i in range((loc_padding * (loc_y - 1)), (loc_padding * loc_y)): # NOQA
                for x_i in range((loc_padding * (loc_x - 1)), (loc_padding * loc_x)): # NOQA
                    image_and_label[image_index][y_i][x_i] = (status, timestamp) # NOQA

    result_table_106 = pd.DataFrame([], columns=['path', 'created_at'])
    result_table_10 = pd.DataFrame([], columns=['path', 'created_at'])

    shutil.rmtree('./dataset/mind_106', ignore_errors=True)
    os.makedirs('dataset/mind_106', exist_ok=True)
    os.makedirs('dataset/mind_106/0', exist_ok=True)
    os.makedirs('dataset/mind_106/1', exist_ok=True)
    os.makedirs('dataset/mind_106/2', exist_ok=True)
    os.makedirs('dataset/mind_106/3', exist_ok=True)

    shutil.rmtree('./dataset/mind_10', ignore_errors=True)
    os.makedirs('dataset/mind_10', exist_ok=True)
    os.makedirs('dataset/mind_10/0', exist_ok=True)
    os.makedirs('dataset/mind_10/1', exist_ok=True)
    os.makedirs('dataset/mind_10/2', exist_ok=True)
    os.makedirs('dataset/mind_10/3', exist_ok=True)

    list_images = glob.glob("original_images/*.jpg")
    for image_i in list_images:
        result_table_106 = split_image(
            image_i, image_and_label, result_table_106, 'mind_106'
        )

        if image_i[16:23] in target_for_10:
            result_table_10 = split_image(
                image_i, image_and_label, result_table_10, 'mind_10'
            )

    result_table_106.to_csv('dataset/mind_106-label_order.csv')
    result_table_10.to_csv('dataset/mind_10-label_order.csv')


def mind_106_amt():
    project_name = 'pre_cyber_disaster_drill3'
    # project_name = 'pre_cyber_disaster_drill'

    images = get_data_from_crowd4u(project_name, 'Image')
    answers = get_data_from_crowd4u(project_name, 'Answer')
    print('images', len(images), images[0])
    print('answers', len(answers), answers[0])

    list_image_and_label = []

    sorted_answers = sorted(answers, key=sort_function)
    for key, image_i in groupby(sorted_answers, key=sort_function):
        status_ids = []
        list_timestamp = []

        for image_i_answer_j in image_i:
            timestamp = image_i_answer_j['answered_at']

            if not timestamp:
                timestamp = image_i_answer_j['updated_at']

            # print(timestamp)
            if timestamp.startswith('2019-10-06') or timestamp.startswith('2019-10-08'): # NOQA
                # print(image_i_answer_j)
                status_ids.append(image_i_answer_j['status_id'])
                list_timestamp.append(timestamp)

        if len(status_ids) > 6:
            # print(key, status_ids, list_timestamp)
            the_status_id = most_frequent(status_ids)
            the_timestamp = int(time.mktime(parser.parse(
                list_timestamp[0]
            ).timetuple()))

            timelapse_start = 1570320000
            timelapse_end = 1570525200

            in_time = (timelapse_start < the_timestamp and timelapse_end > the_timestamp ) # NOQA

            if the_status_id != 3 and in_time:
                the_image = next(filter(
                    lambda x: x['image_id'] == key, images
                ))
                # print(
                # key, the_status_id, list_timestamp[0], the_image['image_url']
                # )

                list_image_and_label.append((
                    the_image['image_url'],
                    the_status_id,
                    the_timestamp
                ))

    # print(list_image_and_label)
    image_and_label = {}

    for image_url, status, timestamp in list_image_and_label:
        image_name = image_url.replace(
            "projects.crowd4u.org/mind/img/{}/dest/".format(project_name), ''
        ).replace('.png', '')

        image_index = image_name[0:4]

        nest_level, loc_y, loc_x = image_name.split('-qv_')[1].split('_')
        nest_level = int(nest_level)
        loc_y = int(loc_y)
        loc_x = int(loc_x)
        loc_padding = int(32 / (2 ** (nest_level - 1)))
        # print(nest_level, loc_y, loc_x)

        if image_index not in image_and_label:
            image_and_label[image_index] = [
                [None for x in range(32)] for y in range(32)
            ]

        if status in [1, 2, 4]:
            for y_i in range((loc_padding * (loc_y - 1)), (loc_padding * loc_y)): # NOQA
                for x_i in range((loc_padding * (loc_x - 1)), (loc_padding * loc_x)): # NOQA
                    image_and_label[image_index][y_i][x_i] = (status, timestamp) # NOQA

    result_table_106 = pd.DataFrame([], columns=['path', 'created_at'])
    result_table_10 = pd.DataFrame([], columns=['path', 'created_at'])

    shutil.rmtree('./dataset/mind_106_amt', ignore_errors=True)
    os.makedirs('dataset/mind_106_amt', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/0', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/1', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/2', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/3', exist_ok=True)

    shutil.rmtree('./dataset/mind_10_amt', ignore_errors=True)
    os.makedirs('dataset/mind_10_amt', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/0', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/1', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/2', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/3', exist_ok=True)

    list_images = glob.glob("original_images/*.jpg")
    for image_i in list_images:
        # result_table_106 = split_image(
        #     image_i, image_and_label, result_table_106, 'mind_106_amt'
        # )

        if image_i[16:23] in target_for_10:
            result_table_10 = split_image(
                image_i, image_and_label, result_table_10, 'mind_10_amt'
            )

    result_table_106.to_csv('dataset/mind_106_amt-label_order.csv')
    result_table_10.to_csv('dataset/mind_10_amt-label_order.csv')


def mind_106_amt2():
    project_name = 'pre_cyber_disaster_drill3'
    # project_name = 'pre_cyber_disaster_drill'

    images = get_data_from_crowd4u(project_name, 'Image')
    answers = get_data_from_crowd4u(project_name, 'Answer')
    print('images', len(images), images[0])
    print('answers', len(answers), answers[0])

    list_image_and_label = []

    sorted_answers = sorted(answers, key=sort_function)
    for key, image_i in groupby(sorted_answers, key=sort_function):
        status_ids = []
        list_timestamp = []

        for image_i_answer_j in image_i:
            timestamp = image_i_answer_j['answered_at']

            if not timestamp:
                timestamp = image_i_answer_j['updated_at']

            # print(timestamp)
            # if timestamp.startswith('2019-10-06') or timestamp.startswith('2019-10-08'): # NOQA
                # print(image_i_answer_j)
            status_ids.append(image_i_answer_j['status_id'])
            list_timestamp.append(timestamp)

        if len(status_ids) > 6:
            # print(key, status_ids, list_timestamp)
            the_status_id = most_frequent(status_ids)
            the_timestamp = int(time.mktime(parser.parse(
                list_timestamp[0]
            ).timetuple()))

            # timelapse_start = 1570320000
            # timelapse_end = 1570525200

            # in_time = (timelapse_start < the_timestamp and timelapse_end > the_timestamp ) # NOQA

            if the_status_id != 3:
                the_image = next(filter(
                    lambda x: x['image_id'] == key, images
                ))
                # print(
                # key, the_status_id, list_timestamp[0], the_image['image_url']
                # )

                list_image_and_label.append((
                    the_image['image_url'],
                    the_status_id,
                    the_timestamp
                ))

    print(list_image_and_label)
    image_and_label = {}

    for image_url, status, timestamp in list_image_and_label:
        image_name = image_url.replace(
            "projects.crowd4u.org/mind/img/{}/dest/".format(project_name), ''
        ).replace('.png', '')

        image_index = image_name[0:4]

        print(image_index)

        nest_level, loc_y, loc_x = image_name.split('-qv_')[1].split('_')
        nest_level = int(nest_level)
        loc_y = int(loc_y)
        loc_x = int(loc_x)
        loc_padding = int(32 / (2 ** (nest_level - 1)))

        loc_padding2 = 2 ** (6-nest_level)
        print(nest_level, loc_y, loc_x, loc_padding, loc_padding2)

        if image_index not in image_and_label:
            image_and_label[image_index] = [
                [None for x in range(32)] for y in range(32)
            ]

        if status in [1, 2, 4]:
            print(range((loc_padding * (loc_y - 1)), (loc_padding * loc_y)))
            for y_i in range((loc_padding * (loc_y - 1)), (loc_padding * loc_y)): # NOQA
                for x_i in range((loc_padding * (loc_x - 1)), (loc_padding * loc_x)): # NOQA
                    image_and_label[image_index][y_i][x_i] = (status, timestamp) # NOQA

    result_table_106 = pd.DataFrame([], columns=['path', 'created_at'])
    result_table_10 = pd.DataFrame([], columns=['path', 'created_at'])

    shutil.rmtree('./dataset/mind_106_amt', ignore_errors=True)
    os.makedirs('dataset/mind_106_amt', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/0', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/1', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/2', exist_ok=True)
    os.makedirs('dataset/mind_106_amt/3', exist_ok=True)

    shutil.rmtree('./dataset/mind_10_amt', ignore_errors=True)
    os.makedirs('dataset/mind_10_amt', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/0', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/1', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/2', exist_ok=True)
    os.makedirs('dataset/mind_10_amt/3', exist_ok=True)

    list_images = glob.glob("original_images/*.jpg")
    for image_i in list_images:
        result_table_106 = split_image(
            image_i, image_and_label, result_table_106, 'mind_106_amt'
        )

        if image_i[16:23] in target_for_10:
            result_table_10 = split_image(
                image_i, image_and_label, result_table_10, 'mind_10_amt'
            )

    result_table_106.to_csv('dataset/mind_106_amt-label_order.csv')
    result_table_10.to_csv('dataset/mind_10_amt-label_order.csv')


if __name__ == "__main__":
    mind_106()
    mind_106_amt2()

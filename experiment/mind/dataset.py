import torchvision
from torchvision.transforms import ToTensor
# from torch.utils.data import Subset

from hactap.utils import ImageFolderWithPaths

height = 122
width = 110

MIND_106_DATASET_PATH = 'dataset/mind_106'
MIND_106_AMT_DATASET_PATH = 'dataset/mind_106_amt'

MIND_10_DATASET_PATH = 'dataset/mind_10'
MIND_10_AMT_DATASET_PATH = 'dataset/mind_10_amt'


def mind_106():
    return ImageFolderWithPaths(
        MIND_106_DATASET_PATH,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((height, width)),
            ToTensor()
        ])
    )


def mind_106_amt():
    return ImageFolderWithPaths(
        MIND_106_AMT_DATASET_PATH,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((height, width)),
            ToTensor()
        ])
    )


def mind_10():
    return ImageFolderWithPaths(
        MIND_10_DATASET_PATH,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((height, width)),
            ToTensor()
        ])
    )


def mind_10_amt():
    return ImageFolderWithPaths(
        MIND_10_AMT_DATASET_PATH,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((height, width)),
            ToTensor()
        ])
    )


# def pickup_ten_images(dataset):
#     target_for_ten = []

#     for index in range(len(dataset)):
#         path, label = dataset.get_label(index)
#         image_id = path.split('/')[3][0:7]

#         if image_id in target_for_10:
#             target_for_ten.append(index)
#         # print(image_id)
#     print(len(target_for_ten))
#     return target_for_ten


def summary(dataset):
    labels = {}

    for index in range(len(dataset)):
        path, label = dataset.get_label(index)

        if label not in labels:
            labels[label] = []

        labels[label].append(label)

    print("All={}, 0={}, 1={}, 2={}, 3={}".format(
        len(dataset),
        len(labels[0]),
        len(labels[1]),
        len(labels[2]),
        len(labels[3])
    ))


def main():
    print('mind_106')
    summary(mind_106())

    print('mind_106_amt')
    summary(mind_106_amt())

    print('mind_10')
    summary(mind_10())

    print('mind_10_amt')
    summary(mind_10_amt())


if __name__ == "__main__":
    main()

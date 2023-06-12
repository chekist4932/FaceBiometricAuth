# import torch
# from torchvision import models, datasets
# import torch.nn as nn
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_classes = 46
#
# test = torchvision.datasets.ImageFolder(os.path.join(dataset_dir, 'test'), transform=transform)
# testset = DataLoader(train, batch_size=15, shuffle=True)
#
# model = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
# model.to(device)
# model.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, num_classes).to(device)
# model.load_state_dict(torch.load('models/model.pth'))
#
#
#
#
#
#
#
#
#
#
#

# import os
# import random
# import shutil
#
#
# def get_random_unique(target_array: list, count: int):
#     temp_counter = 0
#     result = []
#     while temp_counter < count:
#         item = random.choice(target_array)
#         if item not in result:
#             result.append(item)
#             temp_counter += 1
#         else:
#             continue
#     return result
#
#
# dir_on = 'dataset\\train'
# dir_to = 'dataset\\test'
#
# class_folders = os.listdir(dir_on)
# for class_ in class_folders:
#
#     path_to = os.path.join(dir_to, class_)
#     ph = os.path.join(dir_on, class_)
#
#     images = os.listdir(ph)
#     counter = int(len(images) * 0.2)
#     image_names = get_random_unique(images, counter)
#     for image_name in image_names:
#         photo_name = os.path.join(ph, image_name)
#         place_to = os.path.join(dir_to, class_, image_name)
#         shutil.move(photo_name, place_to)
#         print(f'Action: {photo_name} --> {place_to}')

# for i in range(46):
#     os.mkdir(os.path.join(dir_, f"class{i}"))

#
# prerender_dataset_frontal_dir = r"dataset\union"
#
# photos = os.listdir(prerender_dataset_frontal_dir)
#
# for photos_name in photos:
#     tags = photos_name.split(".")
#     print(tags)
#     if len(tags) == 2:
#         continue
#     last_name = prerender_dataset_frontal_dir + '\\' + photos_name
#     new_name = prerender_dataset_frontal_dir + "\\" + tags[0] + '.' + tags[2]
#     os.rename(last_name, new_name)

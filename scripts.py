import os
import random
import shutil


def rename_folders():
    dir = 'dataset\\test'
    folders = os.listdir(dir)
    for id_, folder in enumerate(folders):
        old_path = os.path.join(dir, folder)
        new_path = os.path.join(dir, folder[5:])
        os.rename(old_path, new_path)


def get_random_unique(target_array: list, count: int):
    temp_counter = 0
    result = []
    while temp_counter < count:
        item = random.choice(target_array)
        if item not in result:
            result.append(item)
            temp_counter += 1
        else:
            continue
    return result


def dataset_split():
    dir_on = 'dataset\\train'
    dir_to = 'dataset\\test'

    class_folders = os.listdir(dir_on)
    for class_ in class_folders:

        path_to = os.path.join(dir_to, class_)
        ph = os.path.join(dir_on, class_)

        images = os.listdir(ph)
        counter = int(len(images) * 0.2)
        image_names = get_random_unique(images, counter)
        for image_name in image_names:
            photo_name = os.path.join(ph, image_name)
            place_to = os.path.join(dir_to, class_, image_name)
            shutil.move(photo_name, place_to)
            print(f'Action: {photo_name} --> {place_to}')


def create_folders(dir_: str):
    for i in range(46):
        os.mkdir(os.path.join(dir_, f"class{i}"))





# from tqdm.auto import trange, tqdm
# from time import sleep
#
# for i in range(10):
#     with tqdm(total=100, position=0, desc='text') as progress:
#         for i in range(100):
#             sleep(0.1)
#             progress.set_description(f"temp_val: {i * random.randint(52,798)}")
#             # progress.update()
#         progress.set_description(f"Total: {10000}")

import torch, json, math, matplotlib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from os import listdir, system
from os.path import isfile, join
import numpy as np
from PIL import Image
import random

DATA_PATH = 'liquid_data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != "cpu":
    matplotlib.use('pdf')


shapes = ["cube", "sphere", "cylinder"]
colors = ["gray", "blue", "brown", "yellow", "red", "green", "purple", "cyan"]
shapes_and_colors = [c + ' ' + s for s in shapes for c in colors]
objects_code = {k: i for i, k in enumerate(shapes_and_colors)}
loc_rel = {i: k for i, k in enumerate(['left', 'above', 'right', 'below'])}


# preprocessing
import torchvision
print(torchvision.__version__)
# from torchvision.transforms.autoaugment import RandAugment
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


BICUBIC = Image.BICUBIC
def _convert_image_to_rgb(image):
    return image.convert("RGB")
preprocessing = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        # RandAugment(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class CustomCLEVRImageDataset(Dataset):
    def __init__(self, size, data_path=DATA_PATH, use_random_location_of_main_index=False):
        self.data_path = data_path
        self.json_paths = [f for f in listdir(join(self.data_path, "label")) if
                           isfile(join(self.data_path, "label", f))][: size]
        self.shuffle_indexes = np.random.permutation(len(self.json_paths))
        self.size = size
        if size > len(self.json_paths):
            print(f'There is no enough data for dataset in size {size}')

        self.use_random_location_of_main_index = use_random_location_of_main_index

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        json_name = self.json_paths[self.shuffle_indexes[idx]]
        label_path = join(self.data_path, "label", json_name)

        with open(label_path, 'r') as j:
            label = json.loads(j.read())
        img_path = join(self.data_path, "images", label['image_filename'])

        # img = np.fromfile(img_path, dtype=np.int).reshape(512, 512, 3)
        img = Image.open(img_path)

        # img = np.array(img)[:, :, :3]
        img = preprocessing(img)  #.permute(1, 2, 0)
        # img = torch.tensor(img).float()

        item = self.prepare_item(label)
        # plt.imshow(img)
        # plt.show()
        # self.show_coords(label, Image.open(img_path))
        # print(label['objects'][item[1][2]]['shape'], label['objects'][item[1][2]]['color'], item[0][-1], label['objects'][(item[1][2] + 1) % 2]['shape'], label['objects'][(item[1][2] + 1) % 2]['color'])
        indexes = torch.cat(item[1])
        x, y = (img, indexes), item[0]
        return x, y

    def prepare_item(self, item):
        # make one vector for model output + order of objects for reffering expression + original label
        # the gt y is of size 5 when each entry relates to a different relation
        if self.use_random_location_of_main_index:
            random_obg = random.randint(0, 1)  # randomly choose who is the obj and subj from the two possible object
        else:
            random_obg = 1
        obj_1 = item['objects'][random_obg]
        obj_2 = item['objects'][(random_obg + 1) % 2]
        name_1 = item['objects'][random_obg]['color'] + ' ' + item['objects'][random_obg]['shape']
        name_2 = item['objects'][(random_obg + 1) % 2]['color'] + ' ' + item['objects'][(random_obg + 1) % 2]['shape']
        ndx_1 = torch.nn.functional.one_hot(torch.tensor(objects_code[name_1]), num_classes=len(objects_code))
        ndx_2 = torch.nn.functional.one_hot(torch.tensor(objects_code[name_2]), num_classes=len(objects_code))

        greater_than = 2 if (obj_1['size'] == 'large' and obj_2['size'] == 'small') else (1 if (obj_2['size'] == 'large' and obj_1['size'] == 'small') else 0) # 2 if a is bigger, 1 if the opsite else (equal) 0

        material_relation = int(obj_1['material'] != 'rubber' and obj_2['material'] == 'rubber')  # binary -a is more sparkly than b

        relative_location = self.calc_location_relation(obj_1['pixel_coords'], obj_2['pixel_coords'])  # 0-3 by angle

        higher_than = int(obj_1['3d_coords'][2] > obj_2['3d_coords'][2])  # binary

        closer_than = int(obj_1['pixel_coords'][2] < obj_2['pixel_coords'][2])  # binary

        flow_relation = 0  # 0 no flow, 1 flow between a to b, 2 the upside direction, 3 and 4 for viscosity
        if 'liquid_params' in item:
            if item['liquid_params']['viscosity'] == 0:  # water
                flow_relation = 1 if obj_1['liquid_src'] else 2
            elif item['liquid_params']['viscosity'] == 0.05:  # high viscosity
                flow_relation = 3 if obj_1['liquid_src'] else 4
            else:
                print('error - wired viscosity - ', item['liquid_params']['viscosity'])

        location1, location2 = [obj_1['pixel_coords'][0], obj_1['pixel_coords'][1]], [obj_2['pixel_coords'][0], obj_2['pixel_coords'][1]]
        return np.array([greater_than, higher_than, material_relation, relative_location, flow_relation, closer_than]), (ndx_1 / 24 - 0.5, ndx_2 / 24 - 0.5, torch.Tensor([random_obg-0.5]), torch.Tensor(location1), torch.Tensor(location2))
        #       amounts of options:  c  #18 options

    @staticmethod
    def show_coords(label, image):
        obj_coords = [obj['pixel_coords'] for obj in label['objects']]
        im = np.array(image)[:, :, :3]
        for coords in obj_coords:
            x, y = coords[0], coords[1]
            im[y:y + 10, x:x + 10, :] = 200
        plt.imshow(im)
        plt.show()

    @staticmethod
    def calc_location_relation(coords_a, coords_b):
        x_a, y_a, x_b, y_b = coords_a[0], coords_a[1], coords_b[0], coords_b[1]
        d = x_a - x_b, y_a - y_b
        theta = math.atan2(d[1], d[0])
        if theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
            return 0  # left
        elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            return 1  # above
        elif -math.pi / 4 <= theta < math.pi / 4:
            return 2  # right
        elif math.pi / 4 <= theta < 3 * math.pi / 4:
            return 3  # 'below'


if __name__ == '__main__':
    a = CustomCLEVRImageDataset(size=10)
    for i in range(10):
        b = a.__getitem__(i)
import torch, json, math, matplotlib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from os import listdir, system
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
import clip
from PIL import Image
import itertools, random, copy
from pathlib import Path

DATA_PATH = 'liquid_data'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if str(device) != "cpu":
    matplotlib.use('pdf')

model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("RN50", device=device)
model.eval()


class CustomCLEVRImageDataset(Dataset):
    def __init__(self, data_path=DATA_PATH, get_clip_embedding_mode=0):
        self.data_path = data_path
        self.json_paths = [f for f in listdir(join(self.data_path, "label")) if
                           isfile(join(self.data_path, "label", f))]
        self.get_clip_embedding_mode = get_clip_embedding_mode

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx):
        json_name = self.json_paths[idx]
        label_path = join(self.data_path, "label", json_name)

        with open(label_path, 'r') as j:
            label = json.loads(j.read())
        img_path = join(self.data_path, "images", label['image_filename'])

        if self.get_clip_embedding_mode > 0:
            amount_of_objs = len(label['objects'])
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features, amount_of_objs  # return image features and amount of objects
        return img_path, label

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
    def calc_relation(coords_a, coords_b):
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

    @staticmethod
    def calc_relations(label):
        # returns 2d matrix when the lower ractangle are 0s, and the upper is one of 0,1,2,3 htat represetns the relation
        obj_coords = [obj['pixel_coords'] for obj in label['objects']]
        objs = [obj['color'][0] + ' ' + obj['shape'][0] for obj in label['objects']]
        print(objs)
        m = np.zeros((len(obj_coords, ), len(obj_coords))) - 1
        for i, coords in enumerate(obj_coords):
            for j in range(i + 1, len(obj_coords)):
                m[i, j] = CustomCLEVRImageDataset.calc_relation(obj_coords[i], obj_coords[j])
        return m, objs


if  __name__ == '__main__':
    a = CustomCLEVRImageDataset()
    b = a.__getitem__(0)
    print(b)
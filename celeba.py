import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CelebA(data.Dataset):

    @staticmethod
    def attr_names():
        return [
            ('5_o_Clock_Shadow', '短胡子'),
            ('Arched_Eyebrows', '弯眉毛'),
            ('Attractive', '有吸引力'),
            ('Bags_Under_Eyes', '眼袋'),
            ('Bald', '秃顶'),
            ('Bangs', '刘海'),
            ('Big_Lips', '厚嘴唇'),
            ('Big_Nose', '大鼻子'),
            ('Black_Hair', '黑发'),
            ('Blond_Hair', '金发'),
            ('Blurry', '模糊'),
            ('Brown_Hair', '棕发'),
            ('Bushy_Eyebrows', '浓眉'),
            ('Chubby', '胖脸'),
            ('Double_Chin', '双下巴'),
            ('Eyeglasses', '眼镜'),
            ('Goatee', '山羊胡'),
            ('Gray_Hair', '灰白发'),
            ('Heavy_Makeup', '浓妆'),
            ('High_Cheekbones', '高颧骨'),
            ('Male', '男性'),
            ('Mouth_Slightly_Open', '嘴巴微张'),
            ('Mustache', '八字胡'),
            ('Narrow_Eyes', '小眼睛'),
            ('No_Beard', '无胡须'),
            ('Oval_Face', '鹅蛋脸'),
            ('Pale_Skin', '苍白皮肤'),
            ('Pointy_Nose', '尖鼻子'),
            ('Receding_Hairline', '发迹线后移'),
            ('Rosy_Cheeks', '红润皮肤'),
            ('Sideburns', '连鬓胡'),
            ('Smiling', '微笑'),
            ('Straight_Hair', '直发'),
            ('Wavy_Hair', '卷发'),
            ('Wearing_Earrings', '戴耳环'),
            ('Wearing_Hat', '戴帽子'),
            ('Wearing_Lipstick', '涂口红'),
            ('Wearing_Necklace', '戴项链'),
            ('Wearing_Necktie', '戴领带'),
            ('Young', '年轻')
        ]

    def __init__(self, root, ann_file, transform=None, target_transform=None, loader=default_loader):
        images = []
        targets = []
        
        for line in open(os.path.join(root, ann_file), 'r'):
            sample = line.split()
            if len(sample) != 41:
                raise(RuntimeError("# Annotated face attributes of CelebA dataset should not be different from 40"))
            images.append(sample[0])
            targets.append([int(i) for i in sample[1:]])
        self.images = [os.path.join(root, 'img_align_celeba', img) for img in images]
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        path = self.images[index]
        sample = self.loader(path)
        target = self.targets[index]
        target = torch.LongTensor(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.images)

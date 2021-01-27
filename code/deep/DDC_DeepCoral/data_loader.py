from torchvision import datasets, transforms
import torch

def load_data(data_folder, batch_size, train, kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]), #按比例重定大小
                transforms.RandomCrop(224), #随机裁剪图片使之有一定大小（这里是224）
                transforms.RandomHorizontalFlip(), #随机左右反转
                transforms.ToTensor(), #image（比如PIL）转tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]), #归一化
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
        }
    #ImageFolder是数据加载，参数为数据路径+处理方式
    #可参见https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-datasets/#imagefolder
    #DataLoader将处理好的data封装成tensor
    #可参见https://pytorch.org/docs/stable/data.html
    data = datasets.ImageFolder(root = data_folder, transform=transform['train' if train else 'test'])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs, drop_last = True if train else False)
    return data_loader
 

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()


def readImageAsTensor(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image)
    return image.to(torch.float)


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def PIL_to_tensor(image):
    image = loader(image)
    return image.to(torch.float)


def draw(img, label):
    img = tensor_to_PIL(img)
    draw = ImageDraw.Draw(img)
    for object in label.detectionObjects:
        draw.rectangle(list(object.boundingBox.tensorBox), outline='red')
    img.show()


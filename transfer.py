from PIL import Image
import os
import argparse

# 指定文件夹路径
# folder_path = "data/train"
# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--folder_path', default='./data/train', type=str)
parser.add_argument('--size', default=256, type=int)

args = parser.parse_args()

# 遍历文件夹中的所有文件
for filename in os.listdir(args.folder_path):
    # 只处理图片文件
    if not filename.endswith(".jpg") and not filename.endswith(".jpeg") and not filename.endswith(".png"):
        continue

    image_path = os.path.join(args.folder_path, filename)
    print(image_path)

    try:
        with Image.open(image_path) as image:
            image = image.resize((args.size, args.size), resample=Image.BILINEAR)
            image.save(image_path)
    except OSError:
        os.remove(image_path)
        print(f"Failed to resize {image_path}, deleting file")
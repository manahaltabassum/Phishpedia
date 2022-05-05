from PIL import Image, ImageOps
import os

def resize_compress(factor):
  square_factor = factor
  os.makedirs('./compressed_images/factor_{}'.format((int)(square_factor**2)), exist_ok=True)

  for img_path in os.listdir('./datasets/Sample_phish1000_crop/'):
    img = Image.open('./datasets/Sample_phish1000_crop/{}'.format(img_path)).convert('RGB')

    img = ImageOps.expand(img, (
              (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
              (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

    width, height = img.size
    # 2  --> 4
    # 2.45  --> 6
    # 2.83  --> 8
    # 3.17  --> 10


    resize_factor = ((int)(width/square_factor), (int)(height/square_factor))

    img1 = img.resize(resize_factor)

    img2 = img1.resize((width, height))

    img2 = img

    img2.save('./compressed_images/factor_{}/{}'.format(int(square_factor**2), img_path))

def compress_jpg():
  q = [0, 5, 10, 15, 20]
  for x in q:
    print(str(x))
    os.makedirs('./compressed_images/jpg_compress_{}'.format(str(x)), exist_ok=True)

    for img_path in os.listdir('./datasets/Sample_phish1000_crop/'):
      # print(img_path)
      img = Image.open('./datasets/Sample_phish1000_crop/{}'.format(img_path)).convert('RGB')
      img_path = img_path.split('.png')[0]

      img = ImageOps.expand(img, (
                (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
                (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

      img.save('./compressed_images/jpg_compress_{}/{}.jpg'.format(str(x), img_path), optimize = True, quality = x)
    
  for x in q:
    print(str(x))
    os.makedirs('./compressed_images/png_compress_{}'.format(str(x)), exist_ok=True)

    for img_path in os.listdir('./compressed_images/jpg_compress_{}'.format(str(x))):
      img = Image.open('./compressed_images/jpg_compress_{}/{}'.format(str(x), img_path)).convert('RGB')
      img_path = img_path.split('.jpg')[0]

      img = ImageOps.expand(img, (
                (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2,
                (max(img.size) - img.size[0]) // 2, (max(img.size) - img.size[1]) // 2), fill=(255, 255, 255))

      img.save('./compressed_images/png_compress_{}/{}.png'.format(str(x), img_path))
  
if __name__ == '__main__':
  # resize_compress(2.45)
  compress_jpg()
import os
import glob
import shutil

# path = '/mnt/disks/data/sim2real/ControlNet/IDD_Segmentation/leftImg8bit/train/115/*/'
path = '/mnt/disks/data/sim2real/ControlNet/IDD_Segmentation/gtFine/val/*/'
result = glob.glob(path+'*.json')
print(result)
for image in result:
    print(image)
    target = "/mnt/disks/data/sim2real/ControlNet/train_images_label"
    if (image.endswith(".json")):
        # shutil.move(image,target)
        filename = image.split("/")[-1]
        shutil.move(image, os.path.join(target, filename))
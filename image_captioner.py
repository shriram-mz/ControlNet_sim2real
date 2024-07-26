import requests
import PIL.Image as Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

text = "a realistic photo of"

image_dir = "/mnt/disks/data/sim2real/ControlNet/train_images"
img_name = os.listdir(image_dir)

out=[]
for img in img_name:
    img_path = f"{image_dir}/{img}"
    img_file = Image.open(img_path).convert('RGB')
    inputs = processor(img_file, text, return_tensors="pt").to("cuda")
    caption = model.generate(**inputs)
    out.append(caption)
print(out)





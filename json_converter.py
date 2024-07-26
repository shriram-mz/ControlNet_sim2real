import os
import json
import PIL.Image as Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

text = "a realistic photo of"
out=[]
true = "/mnt/disks/data/sim2real/ControlNet/train_images"
ground_truth = os.listdir(true)
print(len(ground_truth))
#output = open("sample_json", "w")
#json_file = json.dump({'target':ground_truth},output)
#print(json_file)
seg = "/mnt/disks/data/sim2real/ControlNet/segmented_images"
seg_img = os.listdir(seg)
#print(len(seg_img))
#json_file = json.dump({'source':seg_img}, output)
#output.close()


output = open("controlnet_dataset","w")
for i in range(0,len(ground_truth)):
    seg_name = seg_img[i]
    img_path = f"{true}/{seg_name}"
    true_name = ground_truth[i]
    img_file = Image.open(img_path).convert('RGB')
    inputs = processor(img_file, text, return_tensors="pt").to("cuda")
    caption = model.generate(**inputs)
    prompt = processor.decode(caption[0], skip_special_tokens=True)
    json_file = json.dump({'source':seg_name,'target':true_name,'prompt':prompt}, output)
    output.write('\n')
output.close()




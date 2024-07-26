import json
import cv2
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torchvision
from PIL import Image
from safetensors import safe_open
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pytorch_lightning import callbacks

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/mnt/disks/data/sim2real/ControlNet/controlnet_dataset.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/mnt/disks/data/sim2real/ControlNet/segmented_images/' + source_filename)
        target = cv2.imread('/mnt/disks/data/sim2real/ControlNet/train_images/' + target_filename)
        source_Image = Image.fromarray(source)
        target_Image = Image.fromarray(target)
        transform = torchvision.transforms.Resize((512,512))
        source_resized = transform(source_Image)
        target_resized = transform(target_Image)
        source_final = np.array(source_resized)
        target_final = np.array(target_resized)
        source = cv2.cvtColor(source_final, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target_final, cv2.COLOR_BGR2RGB)
        source = source.astype(np.float32) / 255.0
        
        target = (target.astype(np.float32) / 255) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


dataset = MyDataset()
print(len(dataset))

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

def load_textual_inversion(model,embeddings_file_path):
    # tensors = {}
    embedding_matrix = None
    with safe_open(embeddings_file_path, framework="pt", device=0) as f:
        new_tokens = len(f.keys())

        for k in f.keys():
            # tensors[k] = f.get_tensor(k)
            #state_dict = torch.load(embeddings_file_path)
            if embedding_matrix is None:
                embedding_matrix = f.get_tensor(k)
            else :
               embedding_matrix = torch.cat((embedding_matrix,f.get_tensor(k))) 
            

    update_token_embeddings(model,new_tokens,embedding_matrix)

def update_token_embeddings(model,new_add_tokens , new_embedding_matrix):
    old_embeddings = model.cond_stage_model.transformer.text_model.embeddings.token_embedding
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    new_num_tokens = old_num_tokens + new_add_tokens
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
    n = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
    new_embeddings.weight.data[n:,:] = new_embedding_matrix
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding = new_embeddings
    model.cond_stage_model.transformer.text_model.config.vocab_size = new_num_tokens
    model.cond_stage_model.transformer.text_model.vocab_size = new_num_tokens



# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
load_textual_inversion(model,"/mnt/disks/data/sim2real/ControlNet/output_text_inversion/learned_embeds.safetensors")
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
#checkpoint_callback = callbacks.ModelCheckpoint(dirpath='/mnt/disks/data/sim2real/ControlNet/textual_inversion_ControlNet/')
trainer = pl.Trainer(precision=32, callbacks=[logger], accelerator="auto", default_root_dir="/mnt/disks/data/sim2real/ControlNet/textual_inversion_ControlNet/")


# Train!
trainer.fit(model, dataloader)
#trainer.save_checkpoint("/mnt/disks/data/sim2real/ControlNet/textual_inversion_ControlNet/")
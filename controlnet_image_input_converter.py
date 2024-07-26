from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-VL", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()

query = tokenizer.from_list_format([
    {'image': '/mnt/disks/data/sim2real/ControlNet/abc.jpg'},
    {'text': 'Generate an elaborate caption in English:'},
])
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
tokens = response.split()
term = "bus"
term1 = "sidewalk"
term2 = "person"
term3 = "motorcycle" 
term4 = "car"
term5 = "truck"
term6 = "caravan"
term7 = "train"
term8 = "rail track"
term9 = "animal"
term10 = "trailer"
term11 = "bicycle"

lemmatizer = WordNetLemmatizer()
words = []
char = "English:"
for w in tokens:
    words.append(lemmatizer.lemmatize(w))
if term in words or term3 in words or term4 in words or term5 in words or term6 in words or term10 in words:
    tokens.append("<indian-bus>") 
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term3 in words: 
    tokens.append("<indian-car>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term4 in words or term11 in words: 
    tokens.append("<indian-motorcycle>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term5 in words: 
    tokens.append("<indian-truck>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term6 in words:
    tokens.append("<indian-caravan>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term7 in words:
    tokens.append("<indian-train>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term8 in words:
    tokens.append("<indian-rail track>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term9 in words:
    tokens.append("<indian-animal>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
if term10 in words: 
    tokens.append("<indian-trailer>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)
else:
    tokens.append("<realistic-indian>")
    index = tokens.index(char)
    new_tokens = tokens[index + 1:]
    response = " ".join(new_tokens)

print(response)

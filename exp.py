import torch
import clip
from PIL import Image
import glob
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

logs = []
N = len(glob.glob("/home/victor/wikiart/*/*"))
#N = 3000

if False:

    for i, name in enumerate(glob.glob("/home/victor/wikiart/*/*")[:N]):
        print(i, name)
        image = preprocess(Image.open(name)).unsqueeze(0).to(device)
        logits_image = model.encode_image(image)
        logs.append(logits_image.detach().numpy())

    image_features = torch.cat([torch.tensor(log) for log in logs], dim=0)
    torch.save(image_features, '/home/victor/wikiart/image_features.pt')

image_featuresl = torch.load('/home/victor/wikiart/image_features.pt')


print(image_featuresl.shape)

text = clip.tokenize(
    ["a cubist painting with something made of wool in it"]).to(device)
text_features = model.encode_text(text)

image_features = image_featuresl/image_featuresl.norm(dim=-1, keepdim=True)
text_features = text_features/text_features.norm(dim=-1, keepdim=True)

similarity = (100.0 * image_features @ text_features.T).softmax(dim=0)
print(similarity.shape)
values, indices = similarity[:, 0].topk(5)

names = np.asarray(glob.glob("/home/victor/wikiart/*/*")[:N])
print(values)
print(names[indices.numpy()])

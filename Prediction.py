import cv2
import torch
import torch.nn.functional as F

def pred(img):
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = torch.load('./TrainedModel/VGGnet.pth')
   model.to(device)
   model.eval()
   im = cv2.imread(img)
   im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
   inputs = torch.from_numpy(im).float().permute(2, 0, 1).unsqueeze(0)/255
   inputs = inputs.to(device)
   outputs = model(inputs)
   predvalue = outputs.argmax(1).item()
   probs = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]
   if predvalue == 0:
      preclass = '1st class'
      probs = probs[0]
   else:
      preclass = '2nd class'
      probs = probs[1]
   return preclass, probs






import pandas as pd 
import matplotlib.pyplot as plt 
import torch
from fuvai import YNet  
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np 

csv_path = "Filtered_data.csv"
df = pd.read_csv(csv_path,sep=";")

######################################################################
## MODEL INSTANTIATION
input_channels = 1         
output_channels = 64      
n_classes = 1         

model = YNet(input_channels, output_channels, n_classes)
checkpoint_path = 'fuvai_weights.pt'
model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
model.eval()  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
print("Model loaded")
######################################################################

transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.Resize((224, 224)),                
        transforms.ToTensor(),                
        transforms.Normalize(mean=[0], std=[1]) 
])

# iteration over each row in the dataframe , 4k rows around 
for index, row in df.iterrows():
    print("index : ",index)
    if index == 100: 
        break
    img_name = row["Image_name"]
    plane = row["Plane"]
    patient_id = row["Patient_num"]

    if "00216" in img_name or "00627" in img_name or "00628"in img_name or "00629" in img_name:
        continue

    img_file_path = f"FETAL_PLANES_ZENODO/Images/{img_name}.png"

    img = Image.open(img_file_path).convert('RGB')
    tensor_img = transform(img)              
    input_tensor = tensor_img.unsqueeze(0).unsqueeze(0)

    # INPUT 
    img = input_tensor[0, 0, 0].cpu().detach().numpy()  
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title("Input to the model")
    plt.savefig(f"OUTPUT_2/IMG/{img_name}_{plane}.png")
    plt.axis('off')
    plt.show()
    
    print("Input tensor to the model : ",input_tensor.shape)

    with torch.no_grad():
        classification_output, segmentation_output = model(input_tensor)

    # OUTPUT 
    print("classification output : ",classification_output.shape)
    print("segmentation output : ",segmentation_output.shape)
    print("classification output : ",classification_output)

    # Softmax on the classification output
    # binary thresholding on the segmentation mask
    predicted_index = torch.argmax(F.softmax(classification_output,dim=1))
    print("Actual class of the image : ",plane)
    print("predicted index of class from model : ",predicted_index.item())
    probability_val = F.softmax(classification_output, dim=1)[0, predicted_index.item()].item()

    # segmentation_output = segmentation_output
    print("Max:", segmentation_output.max().item())
    print("Min:", segmentation_output.min().item())

    binary_mask = (segmentation_output > 0.1).float()
    print("Number of foreground pixels:", binary_mask.sum().item())

    seg_img = binary_mask[0, 0].cpu().detach().numpy() 
    plt.figure(figsize=(4, 4))
    plt.imshow(seg_img, cmap='gray')
    plt.title("output of the model")
    # plt.savefig(f"OUTPUT/SEG/{img_name}_{plane}_seg_{probability_val}.png")
    plt.axis('off')
    plt.show()

    seg_img_uint8 = (seg_img * 255).astype(np.uint8)
    Image.fromarray(seg_img_uint8).save(f"OUTPUT_2/SEG/{img_name}_{plane}_seg_{probability_val}.png")





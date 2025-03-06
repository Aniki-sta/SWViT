import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
from vit_model import vit_base_patch16_224_in21k as create_model
from tqdm import tqdm 

NUM_CLASSES = 20 
json_path = ''  # 类别json文件路径
model_path = ''  # 模型权重路径
image_path = r''  # 待预测图片或文件夹路径
excel_path = ''
results_df = pd.DataFrame(columns=["Image Name", "Predicted Class", "Predicted Class Name", "Probability"])

def load_class_indices(json_path):
    assert os.path.exists(json_path), f"File '{json_path}' does not exist."
    with open(json_path, "r") as f:
        return json.load(f)

def load_model(model_weight_path, device):
    model = create_model(num_classes=NUM_CLASSES, has_logits=False).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    return model

def preprocess_image(img_path):
    assert os.path.exists(img_path), f"File '{img_path}' does not exist."
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    data_transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    
    img = data_transform(img) 
    return torch.unsqueeze(img, dim=0)

def predict(model, img, class_indict, device):
    with torch.no_grad(): 
        output = model(img.to(device)) 
        if isinstance(output, tuple):
            output = output[0] 
        output = torch.squeeze(output).cpu()  
        predict_probs = torch.softmax(output, dim=0)  
        predict_class = torch.argmax(predict_probs).numpy() 

    return predict_probs, predict_class

def display_results(predict_probs, predict_class, class_indict, img_name):
    predict_class_str = str(predict_class)
    predicted_class_name = class_indict.get(predict_class_str, "Unknown") 
    probability = predict_probs[predict_class].numpy()
    print_res = f"Image: {img_name}, Class: {predicted_class_name}, Prob: {probability:.3f}"
    results_df.loc[len(results_df)] = [img_name, predict_class, predicted_class_name, probability]

def process_image_or_folder(path, model, class_indict, device):
    if os.path.isfile(path):
       
        img_name = os.path.basename(path)
        try:
            img = preprocess_image(path)
            predict_probs, predict_class = predict(model, img, class_indict, device)
            display_results(predict_probs, predict_class, class_indict, img_name)
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    elif os.path.isdir(path):
        image_files = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                img_path = os.path.join(root, filename)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_files.append(img_path)
        
        for img_path in tqdm(image_files, desc="Processing images", ncols=100):
            img_name = os.path.basename(img_path)
            try:
                img = preprocess_image(img_path)
                predict_probs, predict_class = predict(model, img, class_indict, device)
                display_results(predict_probs, predict_class, class_indict, img_name)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
    else:
        raise ValueError(f"Path '{path}' is neither a file nor a directory.")

def save_results_to_excel(output_path):
    global results_df
    if not results_df.empty:
        results_df.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    class_indict = load_class_indices(json_path)
    model = load_model(model_path, device)
    process_image_or_folder(image_path, model, class_indict, device)
    output_excel_path = excel_path
    save_results_to_excel(output_excel_path)

if __name__ == '__main__':
    main()

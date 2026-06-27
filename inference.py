import os
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def preprocess(image_path, size=256):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    # ⚠️【关键】这里必须与你原训练代码的归一化完全一致
    # 如果你原代码是用 img / 255.0，这里也要保持一致
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./sample_data/images')
    parser.add_argument('--output_dir', type=str, default='./sample_data/predictions')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 仅加载固化后的模型
    model = torch.jit.load("dgl_unet_weights.pt", map_location=device)
    model.eval()
    
    img_files = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg'))]
    
    with torch.no_grad():
        for f in tqdm(img_files):
            input_tensor = preprocess(os.path.join(args.input_dir, f)).to(device)
            output = model(input_tensor)
            
            # 后处理：假设输出是 logit，过sigmoid后二值化
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            pred = (prob > 0.5).astype(np.uint8) * 255
            
            cv2.imwrite(os.path.join(args.output_dir, f), pred)

if __name__ == '__main__':
    main()

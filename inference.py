import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm


from models.Net import DGL_UNet

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Demo for DGL-UNet (Anonymous Submission)")
    parser.add_argument('--weight_path', type=str, default='clean_best_weights.pth', help='Path to the cleaned model weights')
    parser.add_argument('--input_dir', type=str, default='./sample_data/images', help='Directory containing testing images')
    parser.add_argument('--gt_dir', type=str, default='./sample_data/masks', help='Directory containing ground truth masks')
    parser.add_argument('--output_dir', type=str, default='./sample_data/predictions', help='Directory to save predictions')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size')
    return parser.parse_args()

def calculate_iou_dice(pred, target):
    pred_bin = (pred > 0).astype(np.uint8)
    target_bin = (target > 0).astype(np.uint8)
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    iou = intersection / (union + 1e-6)
    dice = (2. * intersection) / (pred_bin.sum() + target_bin.sum() + 1e-6)
    return iou, dice

def preprocess_image(image_path, img_size):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img, (img_size, img_size))
    
    
    mean = 159.922
    std = 25.748
    
  
    img_normalized = (img - mean) / std
    
   
    img_min = np.min(img_normalized)
    img_max = np.max(img_normalized)
    
    img_normalized = ((img_normalized - img_min) / (img_max - img_min + 1e-8)) * 255.0
    
   
    img = np.transpose(img_normalized, (2, 0, 1))
    
    return torch.from_numpy(img).unsqueeze(0)

def main():
    print("="*60)
    print("🚀 DGL-UNet Anonymous Inference Demo")
    print("="*60)
    
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference on device: {device}")
    
   
    print("[*] Initializing DGL-UNet architecture...")
    model = DGL_UNet(out_channels=[64, 128, 256, 384, 512])
    
   
    print(f"[*] Loading pre-trained weights from {args.weight_path}...")
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"❌ Cannot find weights file: {args.weight_path}")
        
    model.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    
   
    img_names = [f for f in os.listdir(args.input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(img_names) == 0:
        print(f"❌ No images found in {args.input_dir}")
        return
        
    total_iou, total_dice = 0.0, 0.0
    eval_count = 0
    
  
    print("[*] Starting inference and evaluation...")
    with torch.no_grad():
        for img_name in tqdm(img_names, desc="Processing"):
            img_path = os.path.join(args.input_dir, img_name)
            gt_path = os.path.join(args.gt_dir, img_name) 
            
            input_tensor = preprocess_image(img_path, args.img_size).to(device)
            
           
            output = model(input_tensor)
            
          
            if isinstance(output, (list, tuple)):
                output = output[0]  # 💡 如果出图不对，可以尝试改成 output[-1]
            
          
            prob = output.squeeze().cpu().numpy()
            
            
            pred_mask = (prob > 0.5).astype(np.uint8) * 255
           
            save_path = os.path.join(args.output_dir, img_name)
            cv2.imwrite(save_path, pred_mask)
            
            
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_mask = cv2.resize(gt_mask, (args.img_size, args.img_size))
                    gt_mask = (gt_mask > 127).astype(np.uint8) # 保证GT也是二值化的
                    
                    iou, dice = calculate_iou_dice(pred_mask, gt_mask)
                    total_iou += iou
                    total_dice += dice
                    eval_count += 1
                
    if eval_count > 0:
        print("\n" + "="*60)
        print("📊 Evaluation Results (Sample Data):")
        print(f"Average IoU:  {total_iou / eval_count:.4f}")
        print(f"Average Dice: {total_dice / eval_count:.4f}")
        print("="*60)
    
    print(f"✅ All predictions saved to: {args.output_dir}")

if __name__ == '__main__':
    main()

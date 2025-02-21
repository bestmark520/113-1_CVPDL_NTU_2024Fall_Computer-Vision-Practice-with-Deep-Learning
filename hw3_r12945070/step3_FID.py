import os
import json
import torch
import numpy as np
from scipy import linalg
from PIL import Image
from torchvision.models import inception_v3
from torchvision.transforms import transforms


def calculate_fid_score(original_images, generated_images):
    # 載入 Inception V3 模型
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    # 設定圖像轉換
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 準備特徵提取函數
    def extract_features(images):
        features = []
        for img in images:
            # 載入並轉換圖像
            img_tensor = transform(img).unsqueeze(0)

            # 提取特徵
            with torch.no_grad():
                feature = inception_model(img_tensor)
                feature = feature.squeeze().numpy()

            features.append(feature)

        return np.array(features)

    # 提取原始和生成圖像的特徵
    original_features = extract_features(original_images)
    generated_features = extract_features(generated_images)

    # 計算均值和協方差
    mu1, sigma1 = original_features.mean(axis=0), np.cov(original_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # 計算 FID 分數
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid


def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path).convert('RGB')
            images.append(img)
    return images


def main():
    # 直接指定路徑
    original_image_dir = "images_512x512"
    generated_image_dir1 = "result/text/prompt_w_label"
    generated_image_dir2 = "result/text/prompt_w_suffix"
    generated_image_dir3 = "result/prompt_w_suffix"
    output_file = "hw3_cvpdl_2024_FID.txt"
    input_file = "result/blip2/blip2-opt-2.7b.json"  # 計算FID好像用不到

    # 載入原始圖像
    original_images = load_images(original_image_dir)

    # 載入生成的圖像
    generated_images1 = load_images(generated_image_dir1)
    generated_images2 = load_images(generated_image_dir2)
    generated_images3 = load_images(generated_image_dir3)

    # 計算 FID 分數
    fid_score1 = calculate_fid_score(original_images, generated_images1)
    fid_score2 = calculate_fid_score(original_images, generated_images2)
    fid_score3 = calculate_fid_score(original_images, generated_images3)

    # 準備結果
    fid_results = {
        "generated_text_fid": float(fid_score1),
        "prompt_w_label_fid": float(fid_score2),
        "prompt_w_suffix_fid": float(fid_score3)
    }

    # 將結果寫入文本文件
    with open(output_file, 'w') as f:
        f.write("FID Scores:\n")
        f.write(f"Generated Text FID: {fid_score1}\n")
        f.write(f"Prompt w/ Label FID: {fid_score2}\n")
        f.write(f"Prompt w/ Suffix FID: {fid_score3}\n")

    # 控制台輸出
    print("FID 分數計算完成，已寫入", output_file)
    print(f"Generated Text FID: {fid_score1}")
    print(f"Prompt w/ Label FID: {fid_score2}")
    print(f"Prompt w/ Suffix FID: {fid_score3}")


if __name__ == '__main__':
    main()
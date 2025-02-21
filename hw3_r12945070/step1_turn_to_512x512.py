import os
from PIL import Image
import json
import cv2
import matplotlib.pyplot as plt

def resize_images(input_folder, output_folder, target_size=(512, 512)):
    """
    將 input_folder 中的所有圖片調整為 target_size，並存儲到 output_folder。

    :param input_folder: 原始圖片資料夾
    :param output_folder: 調整後的圖片存儲資料夾
    :param target_size: 目標圖片大小 (寬, 高)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # 如果輸出資料夾不存在，則創建

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # 檢查是否為圖片檔案
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"跳過非圖片檔案: {filename}")
            continue

        try:
            # 開啟圖片並調整大小
            with Image.open(input_path) as img:
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)  # 使用 LANCZOS 作為重採樣方式

                # 輸出圖片
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f"成功處理圖片: {filename}")

        except Exception as e:
            print(f"處理圖片 {filename} 時出錯: {e}")


# 設置資料夾
input_folder = "images"  # 輸入圖片資料夾
output_folder = "images_512x512"  # 調整後圖片輸出資料夾

# 執行圖片調整
resize_images(input_folder, output_folder)


def draw_bboxes(image_path, json_path):
    """
    根據 JSON 檔案中的 bboxes，繪製在圖片上，並顯示圖片。

    :param image_path: 輸入圖片路徑
    :param json_path: 輸入 JSON 檔案路徑，包含 bboxes 和圖片資訊
    """
    # 載入圖片
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 格式，以便 Matplotlib 顯示

    # 讀取 JSON 檔案
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 遍歷 JSON 中的資料，尋找對應的圖片與 bboxes
    for item in data:
        if item["image"] == image_path.split("/")[-1]:  # 根據圖片檔名匹配
            bboxes = item["bboxes"]
            # 在圖片上畫出 bboxes
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                # 畫矩形框
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

    # 顯示圖片
    plt.imshow(image)
    plt.axis('off')  # 不顯示座標軸
    plt.show()


# 使用範例：
image_path = 'images_512x512/pexels-photo-1216589.jpeg'  # 輸入圖片路徑
json_path = 'label.json'  # 輸入 JSON 檔案路徑
draw_bboxes(image_path, json_path)
'''
image_path = './test_20241206/test.jpeg'  # 輸入圖片路徑
json_path = './test_20241206/test_png.json'  # 輸入 JSON 檔案路徑
draw_bboxes(image_path, json_path)
'''

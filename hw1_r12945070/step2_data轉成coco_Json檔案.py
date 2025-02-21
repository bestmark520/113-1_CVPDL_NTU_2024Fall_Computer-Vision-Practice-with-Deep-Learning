import os
import json
import glob
from PIL import Image
from datetime import datetime

def convert_yolo_to_coco(yolo_labels_dir, output_json_path, image_dir):
    categories = [
        {"id": 0, "name": "Person"},
        {"id": 1, "name": "Ear"},
        {"id": 2, "name": "Earmuffs"},
        {"id": 3, "name": "Face"},
        {"id": 4, "name": "Face-guard"},
        {"id": 5, "name": "Face-mask-medical"},
        {"id": 6, "name": "Foot"},
        {"id": 7, "name": "Tools"},
        {"id": 8, "name": "Glasses"},
        {"id": 9, "name": "Gloves"},
        {"id": 10, "name": "Helmet"},
        {"id": 11, "name": "Hands"},
        {"id": 12, "name": "Head"},
        {"id": 13, "name": "Medical-suit"},
        {"id": 14, "name": "Shoes"},
        {"id": 15, "name": "Safety-suit"},
        {"id": 16, "name": "Safety-vest"}
    ]

    coco_format = {
        "categories": categories,
        "images": [],
        "annotations": []
    }

    annotation_id = 0
    image_id = 0
    total_images = 0

    # 獲取所有圖片的檔案名
    image_filenames = glob.glob(os.path.join(image_dir, "*.*"))

    for image_path in image_filenames:
        image_name = os.path.basename(image_path)

        print(f"正在處理: {image_name}")

        try:
            # 獲取圖片的寬度和高度
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            # 圖片資訊加入到 JSON 的 "images" 區段
            coco_format["images"].append({
                "id": image_id,
                "file_name": image_name,
                "height": img_height,
                "width": img_width,
                "date_captured": datetime.now().isoformat()
            })

            # 獲取對應的 YOLO 標籤檔案
            label_path = os.path.join(yolo_labels_dir, os.path.splitext(image_name)[0] + ".txt")

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        # YOLO 格式：<class_id> <x_center> <y_center> <width> <height>
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        # 將 YOLO 座標轉換為 COCO 的 bounding box 格式 [x_min, y_min, width, height]
                        x_min = (x_center - width / 2) * img_width
                        y_min = (y_center - height / 2) * img_height
                        bbox_width = width * img_width
                        bbox_height = height * img_height

                        # 保留浮點數精度
                        bbox = [x_min, y_min, bbox_width, bbox_height]

                        # bbox 面積
                        area = bbox_width * bbox_height

                        # 標註資訊加入到 JSON 的 "annotations" 區段
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": bbox,
                            "area": area,
                            "iscrowd": 0
                        })
                        annotation_id += 1

            # 更新圖片 ID
            image_id += 1

            # 更新已處理的照片數量並印出
            total_images += 1
            print(f"已處理照片數量: {total_images}\n")

        except Exception as e:
            print(f"處理 {image_name} 時發生錯誤: {str(e)}")

    # 保存為 JSON 檔案
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

# 定義路徑
yolo_labels_dir_train = "HW1_2024_dataset/train2017/labels"
yolo_labels_dir_val = "HW1_2024_dataset/valid2017/labels"
output_json_train = "HW1_2024_dataset/annotations/instances_train2017.json"
output_json_val = "HW1_2024_dataset/annotations/instances_val2017.json"
image_dir_train = "HW1_2024_dataset/train2017/images"
image_dir_val = "HW1_2024_dataset/valid2017/images"

# 執行轉換
convert_yolo_to_coco(yolo_labels_dir_train, output_json_train, image_dir_train)
convert_yolo_to_coco(yolo_labels_dir_val, output_json_val, image_dir_val)
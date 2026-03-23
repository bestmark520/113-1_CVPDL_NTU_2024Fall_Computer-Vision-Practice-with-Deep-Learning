# hw1: Object Detection R12945070

1. Put the Dataset provided by the assignment into the `HW1_2024_dataset` folder.Mail`bestmark520@gmail.com`to get dataset.
2. and load the pre-trained model provided by https://github.com/facebookresearch/detr.git (which needs to be downloaded separately). 

3. Change the model class through the `step1_change_class_num.py` in DETR to adjust the model class to match the 17 classes in the assignment dataset.
4. Modify the Dataset according to `step2_data轉成coco_Json檔案.py` to match the COCO format. 
5. Modify `num_classes = 17` in `models/detr.py`. 
6. Modify `date` in `step3_main2_強制使用GPU.py`.

7. Train the model using the following command:
   ```bash
   python step3_main2_強制使用GPU.py
   ```

8. `[step4_預測valid產出json_用GPU計算.py]` Test the trained model with validation image data and save the results as `output.json`.

9. `[step5_eval_1004.py.py]` Test the results (the test directory is located in `./`).

10. My environment settings:

```
python version = 3.11.5
torch==2.4.1+cu121
torchvision==0.19.1+cu121
transformers==4.44.0
scikit-learn==1.4.2
scipy==1.12.0
pycocotools==2.0.8
tqdm==4.66.2
timm==1.0.9
tensorboard==2.12.3
```

## 實作細節

### 1. 環境設定

- 使用 Python 3.8 + PyTorch
- 根據 DETR 官方實作修改資料路徑與類別數
- 修改 `./datasets/coco.py` 中的資料路徑
- 修改 `./models/detr.py` 中的 `num_classes = 17`

### 2. 資料格式轉換

原始資料集不完全是 COCO 格式，因此我撰寫了轉換腳本，將影像與標註轉為 COCO 格式的 JSON，包含：
- 影像資訊（id、file_name、height、width）
- 標註資訊（id、image_id、category_id、bbox、area、iscrowd）

### 3. 訓練設定

- 總訓練週期（epochs）：132
- 學習率（learning rate）：1e-4
- 批次大小（batch size）：1（受限於 GPU 記憶體）
- 損失函數：DETR 預設的損失函數，包含分類損失與邊界框損失（L1 + GIOU）
- 優化器：AdamW
- 預訓練權重：`detr-r50_17.pth`

### 4. 輸出格式

模型輸出為 JSON 格式，與 COCO 格式一致，便於使用官方評估工具進行驗證：

```json
{
  "id": 0,
  "image_id": 0,
  "category_id": 11,
  "bbox": [1937.001, 271.999, 427.002, 876.0],
  "score": 0.95
}
```

## 驗證集表現

| Epochs | mAP (50-95) | mAP50 | mAP75 |
|--------|-------------|-------|-------|
| 130    | 0.4467      | 0.7122| 0.4665|
| 132    | 0.5054      | 0.7697| 0.5447|
| 134    | 0.5029      | 0.7684| 0.5407|

最佳表現出現在第 132 個 epoch，mAP50 達到 0.7697。

## 視覺化與討論

### 長尾效應

我分析了每個類別的訓練樣本數與其 mAP 表現，發現：
- 訓練樣本數較多的類別（如類別 7、3）通常有較高的 mAP
- 少數類別（如類別 6、14）樣本數少，表現明顯較差
- 這顯示資料不平衡會影響模型表現，未來可考慮使用過採樣或資料增強來改善

## 遇到的問題與解決方式

### 問題1：mAP75 一直為 0

- 初步懷疑是 JSON 格式錯誤，但與助教提供的範例比對後格式正確
- 後來發現是 **學習率過低 + 訓練不足** 導致模型無法收斂
- 解決方式：將學習率從 1e-4 提高到 2.5e-4，並增加訓練週期至 132 後，mAP75 順利上升

### 問題2：資料路徑與類別數設定

- 原始 DETR 程式碼預設為 COCO 的 80 類，需手動修改 `num_classes = 17`
- 資料路徑需指向正確的影像與 JSON 檔案

## 結論

本作業成功使用 DETR 模型完成職業傷害預防相關的物件偵測任務，並在驗證集上取得不錯的表現。透過調整學習率與訓練週期，解決了 mAP75 為 0 的問題。長尾效應的分析也提供了未來改進的方向。

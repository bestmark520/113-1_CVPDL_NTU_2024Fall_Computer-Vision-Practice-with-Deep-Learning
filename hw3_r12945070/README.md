# Computer Vision Practice with Deep Learning Homework 3
* Name: 劉高甸
* ID: R12945070

## Data provided by the teaching assistant

* images (folder)
* label.json
* visualiztion_200.json

## Environments

* Ubuntu 20.04
* GeForce RTX™ 2080 Ti 11G
* Python 3.8
* CUDA 11.8
```sh
   pip install -r requirements.txt
```

## Run
```sh
python step1_turn_to_512x512.py
```
```sh
python step2_generate_2160.py
python step3_FID.py
```
```sh
python step4_generate_200.py
```

## Good luck
### After completing all the programs, you will have 200 images with corresponding bounding boxes to submit for the homework3.


## 整體流程
### Step 1：圖像描述生成（Image Captioning）
#### 要做的事：
- 對 HW1 資料集中的每張圖片，使用 **BLIP-2** 模型自動產生一段文字描述。
- 比較不同 BLIP-2 模型的效果，選擇一個比較好的。

#### 我做了什麼：
- 因為設備有限，我比較了兩個模型：
  - `Salesforce/blip2-opt-2.7b`
  - `Salesforce/blip2-flan-t5-xl`
- 我主觀觀察後覺得 `blip2-opt-2.7b` 產生的句子比較準確，所以選用它。

### Step 2：設計提示模板（Prompt Template）
#### 要做的事：
- 設計不同的提示模板，用來告訴 GLIGEN 要生成怎樣的圖片。
- 模板會結合 BLIP-2 產生的文字描述 + 類別名稱 + 圖片尺寸等資訊。

#### 我設計的模板：
```python
"prompt_w_label": f"{generated_text}, focus on {labels_str}, focus on bboxes, high resolution, highly detailed"
"prompt_w_suffix": f"{generated_text}, professional quality, highly detailed"
```

- 我發現不要加入太複雜的提示，反而有助於生成品質。
- 後面會用 FID 分數來比較哪個模板比較好。

### Step 3：使用 GLIGEN 生成圖片
#### 要做的事：
- 使用 GLIGEN 模型，根據文字提示 + 邊界框來生成圖片。
- 比較兩種生成方式：
  - **Text Grounding**：只用文字提示，不給邊界框。
  - **Layout-to-Image**：文字提示 + 邊界框，讓物件出現在指定位置。

#### 我使用的模型：
- Text Grounding：`masterful/gligen-1-4-generation-text-box`
- Layout-to-Image：`anhnct/Gligen_Text_Image`

### Step 4：計算 FID 分數（評估生成品質）
#### 要做的事：
- 用 **FID（Fréchet Inception Distance）** 來評估生成圖片與真實圖片的相似度。
- FID 越低，表示生成圖片越接近真實圖片。

#### 我的結果：

| 方法 | 模板 | FID |
|------|------|-----|
| Text Grounding | Template #1 (prompt_w_label) | 114.03 |
| Text Grounding | Template #2 (prompt_w_suffix) | 113.45 |
| Layout-to-Image | Template #2 | 113.62 |

#### 我的選擇：
- Text Grounding 中，`prompt_w_suffix` 的 FID 比較低（113.45），所以選它。
- 但 Layout-to-Image 的 FID 反而稍微變高（113.62），我推測是因為加入邊界框的限制，讓生成變得更困難。
- 不過作業要求是要「符合邊界框」，所以我還是選擇用 Layout-to-Image + prompt_w_suffix。

### Step 5：調整參數讓圖片更符合邊界框
#### 我發現的關鍵：
- GLIGEN 有一個參數 `guidance_scale`，範圍是 1～12。
- 值越高，模型越會依照提示和邊界框去生成。
- 我設定為 **11**，這樣生成的圖片能比較精準地符合邊界框的位置和形狀。

### Step 6：視覺化比較
#### 要做的事：
- 把原圖和生成的圖片放在一起比較，確認生成效果。

#### 我觀察到的：
- 生成的圖片大致能維持原圖的場景與物件位置。
- 但細節（如顏色、質感）會有些許差異。
- 整體來說，生成圖片還算符合預期。

# 113-1_CVPDL_NTU_2024Fall_Computer-Vision-Practice-with-Deep-Learning
113-1_鄭文皇_電腦視覺實務與深度學習_CVPDL_NTU_2024Fall_Computer Vision Practice with Deep Learning

## hw1 作業1：物件偵測（職業傷害預防）
- 設計一個物件偵測模型，能夠從 RGB 影像中偵測出與職業傷害預防相關的物件，並輸出每個物件的類別與邊界框。
- 必須使用 **基於 Transformer** 或 **基於 Mamba** 的模型。
- 允許使用任何預訓練權重與資料增強技術。
- 不可使用驗證集進行訓練。
- 我選擇使用 **DETR（Detection Transformer）**，這是一個結合 CNN 與 Transformer 的端到端物件偵測模型，能夠直接輸出最終的偵測結果。

### 資料集
- 訓練集：4319 張影像
- 驗證集：2160 張影像
- 測試集：1620 張影像
- 類別數：17

## HW3：利用 BLIP-2 + GLIGEN 進行資料擴增

HW1 的資料集中，有些類別的圖片數量很少（例如只有幾十張），導致訓練出來的物件偵測模型在這些類別上表現很差。  
這份作業的目的是：

- 使用 **BLIP-2** 自動為每張圖片生成文字描述（caption）。
- 設計不同的 **提示模板（prompt template）**。
- 使用 **GLIGEN** 模型，根據文字描述 + 邊界框（bounding box）來「生成新的圖片」。
- 使用 FID 評估生成圖片品質 比較 Text Grounding 與 Layout-to-Image 兩種方法的優劣
- 用這些生成的圖片來擴增訓練資料，希望提升 HW1 模型的表現。

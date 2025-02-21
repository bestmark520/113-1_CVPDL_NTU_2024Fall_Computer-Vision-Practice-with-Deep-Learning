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

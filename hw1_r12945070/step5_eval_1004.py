import json
import numpy as np

date = '2024'

####
pred_file = f'./output_val_{date}.json'
gt_file = './valid_target.json'


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    iou = intersection / union if union > 0 else 0
    return iou


def calculate_precision_recall(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold):
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)

    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    matched_gt = set()

    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1

        for j, gt_box in enumerate(gt_boxes):
            if gt_labels[j] == pred_labels[i]:  # Only consider the same label
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

        # Debug print to show the IoU and the ground truth matching
        print(f'Prediction {i}: best IoU = {best_iou:.4f}, matched GT = {best_gt_idx}')

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp[i] = 1  # True positive
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1  # False positive

    fn = len(gt_boxes) - len(matched_gt)  # False negatives

    # Debugging precision and recall counts
    print(f'True Positives: {sum(tp)}, False Positives: {sum(fp)}, False Negatives: {fn}')

    return tp, fp, fn


def calculate_ap(tp, fp, fn):
    if len(tp) == 0:
        precision = np.array([0])
        recall = np.array([0])
    else:
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)
        recall = tp_cumsum / (tp_cumsum + fn + np.finfo(float).eps)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap


import numpy as np

def calculate_map_per_instance(pred_data, gt_data, iou_thresholds=None):
    # 如果沒有指定IoU閾值，則使用0.5到0.95之間的標準IoU範圍
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)  # 產生0.5到0.95之間的10個閾值

    # 創建一個字典來儲存每個IoU閾值的AP
    ap_per_iou_threshold = {iou: [] for iou in iou_thresholds}

    # 用來儲存每個實例的平均AP
    aps_per_instance = []

    # 遍歷每個實例的預測數據
    for instance in pred_data:
        # 獲取該實例的預測框和標籤
        pred_boxes = pred_data[instance]['boxes']
        pred_labels = pred_data[instance]['labels']

        # 獲取該實例的真實框和標籤
        gt_boxes = gt_data.get(instance, {}).get('boxes', [])
        gt_labels = gt_data.get(instance, {}).get('labels', [])

        # 如果該實例既沒有真實框也沒有預測框，跳過
        if not gt_boxes and not pred_boxes:
            print(f'Instance {instance} has no ground truth and no predictions, skipping.')
            continue

        # Debug：打印預測框和真實框的數量，確保數據加載正確
        print(f'Instance {instance}: {len(pred_boxes)} pred boxes, {len(gt_boxes)} gt boxes')

        aps = []  # 用來儲存該實例在每個IoU閾值下的AP
        for iou_thresh in iou_thresholds:
            # 計算該閾值下的TP、FP、FN
            tp, fp, fn = calculate_precision_recall(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh)
            # 根據TP、FP、FN計算AP
            ap = calculate_ap(tp, fp, fn)
            aps.append(ap)

            # 將每個IoU閾值下的AP存入字典
            ap_per_iou_threshold[iou_thresh].append(ap)

        # 計算該實例的平均AP並存入列表
        aps_per_instance.append(np.mean(aps))

    # 計算不同IoU閾值下的mAP
    mAP50 = np.mean(ap_per_iou_threshold[0.5]) if ap_per_iou_threshold[0.5] else 0
    mAP75 = np.mean(ap_per_iou_threshold[0.75]) if ap_per_iou_threshold[0.75] else 0

    # 計算0.5到0.95之間的mAP平均值
    mAP_50_95 = np.mean([np.mean(aps) if aps else 0 for aps in ap_per_iou_threshold.values()])

    # 打印結果
    print()
    print(f'mAP50 (across all instances): {mAP50:.4f}')
    print(f'mAP75 (across all instances): {mAP75:.4f}')
    print(f'mAP50-95 (across all instances): {mAP_50_95:.4f}')
    print(f'aps_per_instance is {np.mean(aps_per_instance):.4f}')

    # 返回mAP50, mAP75, mAP50-95以及所有實例的平均AP
    return mAP50, mAP75, mAP_50_95, np.mean(aps_per_instance)

# 假設 calculate_precision_recall 和 calculate_ap 是你已有的輔助函數


'''
def calculate_map_per_instance(pred_data, gt_data, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    aps_per_instance = []
    ap_all = {iou_thresh: [] for iou_thresh in iou_thresholds}  # Store AP for each IoU threshold

    for instance in pred_data:
        pred_boxes = pred_data[instance]['boxes']
        pred_labels = pred_data[instance]['labels']

        gt_boxes = gt_data.get(instance, {}).get('boxes', [])
        gt_labels = gt_data.get(instance, {}).get('labels', [])

        if not gt_boxes or not pred_boxes:
            print(f'Instance {instance} is empty, skipping.')
            continue

        print(f'Instance {instance}: {len(pred_boxes)} pred boxes, {len(gt_boxes)} gt boxes')

        aps = []
        for iou_thresh in iou_thresholds:
            tp, fp, fn = calculate_precision_recall(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh)
            ap = calculate_ap(tp, fp, fn)
            aps.append(ap)

            # Store AP for the specific IoU threshold
            ap_all[iou_thresh].append(ap)

        aps_per_instance.append(np.mean(aps))

    # Calculate and print mAP for each IoU threshold
    for iou_thresh in iou_thresholds:
        mAP = np.mean(ap_all[iou_thresh]) if ap_all[iou_thresh] else 0
        print(f'mAP@{iou_thresh:.2f}: {mAP:.4f}')

    mAP_50_95 = np.mean(aps_per_instance)

    print(f'\nmAP50-95 (across all instances): {mAP_50_95:.4f}')

    return ap_all, mAP_50_95


'''


if __name__ == "__main__":
    # Parse command line arguments
    '''
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate mAP50-95 for object detection.')
    parser.add_argument('pred_file', type=str, help='Path to the prediction JSON file.')
    parser.add_argument('gt_file', type=str, help='Path to the ground truth JSON file.')

    args = parser.parse_args()

    # Load prediction and ground truth JSON files
    pred_json = load_json(args.pred_file)
    gt_json = load_json(args.gt_file)

    指令：
    python step5_eval_1004.py output_val_2024_1005_0430.json valid_target.json

    '''

    # 載入預測和真實標籤資料
    pred_json = load_json(pred_file)
    gt_json = load_json(gt_file)

    calculate_map_per_instance(pred_json, gt_json)
import pandas as pd
from scipy.stats import ttest_ind_from_stats

data = {"Method": ["KAU-Net", "EGE-UNet", "I²U-Net", 'Swin-Unet', 'Trans-Unet', 'H-Net', "DPAC-UNet", "HMRNet",
                   "CPF-Net", "DO-Net", "nnU-Net", "CE-Net", "BCDU-Net", "DenseASPP", "Deeplabv3+", "U-Net++",
                   "Res-UNet", "AttU-Net",
                   "U-Net"],
        "IoU_mean": [85.32, 80.94, 83.66, 82.88, 82.60, 82.76, 82.67, 82.61, 82.51, 82.32, 82.28, 81.94, 81.55,
                     82.47, 82.29, 81.89, 81.85, 81.66, 81.23],
        "IoU_std": [1.63, 0.11, 0.33, 0.46, 0.31, 0.39, 0.32, 0.43, 0.33, 0.42,
                    0.49, 0.53, 0.58, 0.25, 0.28, 0.54, 0.44, 0.59, 0.54],
        "Dice_mean": [93.84, 89.46, 90.10, 89.61, 89.44, 89.49,
                      89.45, 89.42, 89.39, 89.22, 89.26, 88.93, 88.74, 89.36, 89.28, 89.15, 88.84, 88.55, 88.23],
        "Dice_std": [0.87,
                     0.07, 0.35, 0.36, 0.21, 0.35, 0.21, 0.34, 0.23, 0.34, 0.38, 0.32, 0.33, 0.15, 0.21, 0.42, 0.32,
                     0.47,
                     0.42]}  # ISIC2018
n = 518

# data = {
#     "Method": ["KAU-Net", 'Swin-Unet', 'Trans-Unet', 'MRUnet', "MedT", "AttU-Net", "U-Net++", "U-Net"],
#     "IoU_mean": [82.87, 82.06, 80.40, 80.89, 75.47, 80.69, 79.13, 74.78],
#     "IoU_std": [0.12, 0.73, 1.04, 1.67, 3.46, 1.66, 1.70, 1.67],
#     "Dice_mean": [90.12, 89.58, 88.40, 88.73, 85.92, 88.80, 87.56, 85.45],
#     "Dice_std": [0.08, 0.57, 0.74, 1.17, 2.93, 1.07, 1.17, 1.25]
# }  # Glas
# n = 80

# data = {
#     "Method": ["KAU-Net", 'Swin-Unet', 'Trans-Unet', 'MRUnet', "MedT", "AttU-Net", "U-Net++", "U-Net"],
#     "IoU_mean": [63.15, 63.77, 65.05, 64.83, 63.37, 63.47, 62.86, 62.86],
#     "IoU_std": [0.04, 1.15, 1.28, 2.87, 3.11, 1.16, 3.00, 3.00],
#     "Dice_mean": [85.10, 77.69, 78.53, 78.22, 77.46, 76.67, 77.01, 76.54],
#     "Dice_std": [0.03, 0.94, .06, 2.47, 2.38, 1.06, 2.10, 2.62]
# }  # MoNuSeg
# n = 14

df = pd.DataFrame(data)

# 对每个方法进行t检验
results = []
for _, row in df[df["Method"] != "KAU-Net"].iterrows():
    # IoU检验
    t_iou, p_iou = ttest_ind_from_stats(
        mean1=df["IoU_mean"][0], std1=df["IoU_std"][0], nobs1=n,
        mean2=row["IoU_mean"], std2=row["IoU_std"], nobs2=n,
        equal_var=False
    )
    # Dice检验
    t_dice, p_dice = ttest_ind_from_stats(
        mean1=df["Dice_mean"][0], std1=df["Dice_std"][0], nobs1=n,
        mean2=row["Dice_mean"], std2=row["Dice_std"], nobs2=n,
        equal_var=False
    )
    results.append({
        "Comparison": f"KAU-Net vs {row['Method']}",
        "IoU_t": t_iou, "IoU_p": p_iou,
        "Dice_t": t_dice, "Dice_p": p_dice
    })

results_df = pd.DataFrame(results)
print(results_df)

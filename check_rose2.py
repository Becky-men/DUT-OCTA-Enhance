#!/usr/bin/env python3
"""
檢查ROSE-2數據集：判斷是否更適合圖像增強任務
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 配置中文字體支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'PingFang SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

def check_rose2_dataset():
    """檢查ROSE-2數據集的original和gt關係"""
    
    print("="*70)
    print("ROSE-2數據集分析")
    print("="*70)
    
    # 檢查幾個樣本
    sample_files = ['1_OD_SVP.png', '1_OS_SVP.png', '2_OD_SVP.png']
    
    fig, axes = plt.subplots(len(sample_files), 4, figsize=(20, 5*len(sample_files)))
    if len(sample_files) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, filename in enumerate(sample_files):
        original_path = f'Dataset/ROSE/ROSE-2/train/original/{filename}'
        gt_path = f'Dataset/ROSE/ROSE-2/train/gt/{filename}'
        
        if not os.path.exists(original_path):
            print(f"⚠️  找不到文件: {original_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"⚠️  找不到文件: {gt_path}")
            continue
        
        # 讀取圖像
        original = Image.open(original_path).convert('L')
        gt = Image.open(gt_path).convert('L')
        
        original_arr = np.array(original)
        gt_arr = np.array(gt)
        
        print(f"\n【樣本 {idx+1}: {filename}】")
        print(f"  Original:")
        print(f"    尺寸: {original_arr.shape}")
        print(f"    值範圍: [{original_arr.min()}, {original_arr.max()}]")
        print(f"    唯一值數量: {len(np.unique(original_arr))}")
        print(f"    平均值: {original_arr.mean():.2f}")
        
        print(f"  GT:")
        print(f"    尺寸: {gt_arr.shape}")
        print(f"    值範圍: [{gt_arr.min()}, {gt_arr.max()}]")
        print(f"    唯一值數量: {len(np.unique(gt_arr))}")
        print(f"    平均值: {gt_arr.mean():.2f}")
        
        # 判斷GT是否為二值化
        unique_gt = len(np.unique(gt_arr))
        if unique_gt <= 3:
            print(f"  ⚠️  GT是二值化標註（{unique_gt}個唯一值）")
            gt_type = "二值化標註"
        elif unique_gt < 20:
            print(f"  ⚠️  GT是低灰度級圖像（{unique_gt}個唯一值）")
            gt_type = "低灰度級"
        else:
            print(f"  ✅ GT是灰度圖像（{unique_gt}個唯一值）- 可能適合增強任務！")
            gt_type = "灰度圖像"
        
        # 可視化
        axes[idx, 0].imshow(original, cmap='gray')
        axes[idx, 0].set_title(f'{filename}\nOriginal', fontsize=10)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(gt, cmap='gray')
        axes[idx, 1].set_title(f'GT\n({gt_type})', fontsize=10)
        axes[idx, 1].axis('off')
        
        # 差異圖
        diff = np.abs(original_arr.astype(float) - gt_arr.astype(float))
        axes[idx, 2].imshow(diff, cmap='hot')
        axes[idx, 2].set_title('差異圖\n(Original - GT)', fontsize=10)
        axes[idx, 2].axis('off')
        
        # 直方圖對比
        axes[idx, 3].hist(original_arr.ravel(), bins=50, alpha=0.5, label='Original', color='blue')
        axes[idx, 3].hist(gt_arr.ravel(), bins=50, alpha=0.5, label='GT', color='red')
        axes[idx, 3].set_title('像素分佈對比', fontsize=10)
        axes[idx, 3].legend()
        axes[idx, 3].grid(True, alpha=0.3)
    
    plt.suptitle('ROSE-2數據集分析：Original vs GT', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('rose2_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ ROSE-2分析圖已保存到: rose2_analysis.png")
    
    print("\n" + "="*70)

def compare_rose1_rose2():
    """對比ROSE-1和ROSE-2的數據特性"""
    
    print("\n" + "="*70)
    print("ROSE-1 vs ROSE-2 對比")
    print("="*70)
    
    # ROSE-1
    print("\n【ROSE-1 (SVC模式)】")
    rose1_img = Image.open('Dataset/ROSE/ROSE-1/SVC/train/img/01.tif').convert('L')
    rose1_gt = Image.open('Dataset/ROSE/ROSE-1/SVC/train/gt/01.tif').convert('L')
    
    rose1_img_arr = np.array(rose1_img)
    rose1_gt_arr = np.array(rose1_gt)
    
    print(f"  img: 唯一值={len(np.unique(rose1_img_arr))}, 範圍=[{rose1_img_arr.min()}, {rose1_img_arr.max()}]")
    print(f"  gt:  唯一值={len(np.unique(rose1_gt_arr))}, 範圍=[{rose1_gt_arr.min()}, {rose1_gt_arr.max()}]")
    
    if len(np.unique(rose1_gt_arr)) <= 3:
        print("  結論: ROSE-1的gt是二值化標註 ❌ 不適合灰度圖像增強")
    
    # ROSE-2
    print("\n【ROSE-2】")
    if os.path.exists('Dataset/ROSE/ROSE-2/train/original/1_OD_SVP.png'):
        rose2_orig = Image.open('Dataset/ROSE/ROSE-2/train/original/1_OD_SVP.png').convert('L')
        rose2_gt = Image.open('Dataset/ROSE/ROSE-2/train/gt/1_OD_SVP.png').convert('L')
        
        rose2_orig_arr = np.array(rose2_orig)
        rose2_gt_arr = np.array(rose2_gt)
        
        print(f"  original: 唯一值={len(np.unique(rose2_orig_arr))}, 範圍=[{rose2_orig_arr.min()}, {rose2_orig_arr.max()}]")
        print(f"  gt:       唯一值={len(np.unique(rose2_gt_arr))}, 範圍=[{rose2_gt_arr.min()}, {rose2_gt_arr.max()}]")
        
        if len(np.unique(rose2_gt_arr)) > 20:
            print("  結論: ROSE-2的gt是灰度圖像 ✅ 可能適合灰度圖像增強")
        else:
            print("  結論: ROSE-2的gt也是二值化/低灰度級")
    else:
        print("  ⚠️  ROSE-2數據集不存在")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    check_rose2_dataset()
    compare_rose1_rose2()


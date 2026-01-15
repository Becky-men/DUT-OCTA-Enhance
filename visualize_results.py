#!/usr/bin/env python3
"""
可視化對比腳本：檢查GAN輸出是否符合圖像增強需求
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 無GUI模式

# 配置中文字體支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'PingFang SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

def visualize_comparison(result_dir, output_path='comparison_analysis.png'):
    """對比顯示原圖、生成圖、標註圖"""
    
    print(f"正在讀取測試結果從: {result_dir}")
    
    # 讀取第一組測試結果
    real_A = Image.open(f'{result_dir}/01_real_A.png').convert('L')
    fake_B = Image.open(f'{result_dir}/01_fake_B.png').convert('L')
    real_B = Image.open(f'{result_dir}/01_real_B.png').convert('L')
    
    # 創建對比圖
    fig = plt.figure(figsize=(18, 12))
    
    # 第一行：圖像對比
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(real_A, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('輸入: 原始OCTA\n(低質量、有噪聲)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(fake_B, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('輸出: GAN生成結果\n⚠️ 二值化輸出', fontsize=14, fontweight='bold', color='red')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(real_B, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('訓練目標: 標準標註\n(二值化血管標籤)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 第二行：局部放大
    # 選擇中心區域放大
    h, w = np.array(real_A).shape
    center_crop = (slice(h//4, 3*h//4), slice(w//4, 3*w//4))
    
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(np.array(real_A)[center_crop], cmap='gray', vmin=0, vmax=255)
    ax4.set_title('原始OCTA (放大)', fontsize=12)
    ax4.axis('off')
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(np.array(fake_B)[center_crop], cmap='gray', vmin=0, vmax=255)
    ax5.set_title('GAN輸出 (放大)', fontsize=12)
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.imshow(np.array(real_B)[center_crop], cmap='gray', vmin=0, vmax=255)
    ax6.set_title('標註 (放大)', fontsize=12)
    ax6.axis('off')
    
    # 第三行：直方圖分析
    for idx, (img, title, color) in enumerate([
        (real_A, '原始OCTA', 'blue'),
        (fake_B, 'GAN生成', 'red'),
        (real_B, '標註', 'green')
    ]):
        ax = plt.subplot(3, 3, 7 + idx)
        arr = np.array(img)
        ax.hist(arr.ravel(), bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_title(f'{title} 像素分佈', fontsize=12)
        ax.set_xlabel('像素值 (0-255)', fontsize=10)
        ax.set_ylabel('頻率', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 標註統計信息
        unique_vals = len(np.unique(arr))
        stats_text = f'唯一值: {unique_vals}\n'
        stats_text += f'均值: {arr.mean():.1f}\n'
        stats_text += f'標準差: {arr.std():.1f}'
        
        ax.text(0.98, 0.97, stats_text, 
                transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('GAN圖像增強結果分析\n⚠️ 問題：模型輸出二值化標註而非增強的灰度圖', 
                 fontsize=16, fontweight='bold', color='red', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 對比圖已保存到: {output_path}")
    
    # 詳細統計分析
    print("\n" + "="*70)
    print("圖像統計分析")
    print("="*70)
    
    for name, img in [('原始OCTA', real_A), ('GAN生成', fake_B), ('標註GT', real_B)]:
        arr = np.array(img)
        unique_vals = np.unique(arr)
        
        print(f"\n【{name}】")
        print(f"  圖像尺寸: {arr.shape}")
        print(f"  數據類型: {arr.dtype}")
        print(f"  值範圍: [{arr.min()}, {arr.max()}]")
        print(f"  唯一值數量: {len(unique_vals)}")
        print(f"  平均值: {arr.mean():.2f}")
        print(f"  標準差: {arr.std():.2f}")
        print(f"  中位數: {np.median(arr):.2f}")
        
        # 判斷是否為二值化圖像
        if len(unique_vals) <= 3:
            print(f"  ⚠️  這是二值化圖像！唯一值: {unique_vals}")
        elif len(unique_vals) < 20:
            print(f"  ⚠️  灰度級很少，接近二值化！唯一值: {len(unique_vals)}")
        else:
            print(f"  ✅ 這是灰度圖像（{len(unique_vals)}個灰度級）")
    
    print("\n" + "="*70)
    
    return arr

def compare_multiple_results(result_dir, num_samples=3, output_path='multi_comparison.png'):
    """對比多組測試結果"""
    
    print(f"\n正在生成多圖對比...")
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        idx = f"{i+1:02d}"
        
        try:
            real_A = Image.open(f'{result_dir}/{idx}_real_A.png').convert('L')
            fake_B = Image.open(f'{result_dir}/{idx}_fake_B.png').convert('L')
            real_B = Image.open(f'{result_dir}/{idx}_real_B.png').convert('L')
            
            axes[i, 0].imshow(real_A, cmap='gray')
            axes[i, 0].set_title(f'樣本{i+1}: 原始輸入', fontsize=12)
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(fake_B, cmap='gray')
            axes[i, 1].set_title(f'樣本{i+1}: GAN輸出', fontsize=12, color='red')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(real_B, cmap='gray')
            axes[i, 2].set_title(f'樣本{i+1}: 標註目標', fontsize=12)
            axes[i, 2].axis('off')
            
        except Exception as e:
            print(f"  跳過樣本{i+1}: {e}")
    
    plt.suptitle('多樣本對比分析', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 多圖對比已保存到: {output_path}")

if __name__ == '__main__':
    result_dir = 'pytorch-CycleGAN-and-pix2pix/results/rose_svc_pix2pix/test_400/images'
    
    print("="*70)
    print("GAN圖像增強結果分析工具")
    print("="*70)
    
    # 單圖詳細分析
    visualize_comparison(result_dir, output_path='comparison_analysis.png')
    
    # 多圖對比
    compare_multiple_results(result_dir, num_samples=3, output_path='multi_comparison.png')
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)


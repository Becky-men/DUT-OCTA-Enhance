#!/usr/bin/env python3
"""
完整的推理演示腳本：展示OCTA圖像增強效果
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 配置中文字體支持
try:
    # 嘗試使用系統中文字體
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'PingFang SC', 'STHeiti', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
except Exception as e:
    print(f"字體配置警告: {e}")
    # 如果配置失敗，使用英文標籤
    USE_CHINESE = False
else:
    USE_CHINESE = True

def create_side_by_side_comparison(test_results_dir, output_dir='demo_results'):
    """創建並排對比圖"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("OCTA圖像增強效果演示")
    print("="*70)
    
    # 獲取所有測試樣本
    image_files = [f for f in os.listdir(test_results_dir) if f.endswith('_real_A.png')]
    sample_ids = [f.replace('_real_A.png', '') for f in image_files]
    
    print(f"\n找到 {len(sample_ids)} 個測試樣本")
    
    # 為每個樣本創建對比圖
    for idx in sample_ids[:5]:  # 只處理前5個樣本作為演示
        print(f"\n處理樣本 {idx}...")
        
        # 讀取圖像
        real_A_path = os.path.join(test_results_dir, f'{idx}_real_A.png')
        fake_B_path = os.path.join(test_results_dir, f'{idx}_fake_B.png')
        real_B_path = os.path.join(test_results_dir, f'{idx}_real_B.png')
        
        if not all(os.path.exists(p) for p in [real_A_path, fake_B_path, real_B_path]):
            print(f"  跳過：文件不完整")
            continue
        
        real_A = np.array(Image.open(real_A_path).convert('L'))
        fake_B = np.array(Image.open(fake_B_path).convert('L'))
        real_B = np.array(Image.open(real_B_path).convert('L'))
        
        # 創建大型對比圖
        fig = plt.figure(figsize=(20, 12))
        
        # 第一行：完整圖像
        ax1 = plt.subplot(3, 3, 1)
        ax1.imshow(real_A, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'樣本{idx} - 原始OCTA\n(低質量，有噪聲)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 3, 2)
        ax2.imshow(fake_B, cmap='gray', vmin=0, vmax=255)
        ax2.set_title(f'樣本{idx} - GAN增強結果\n對比度提升，血管清晰', 
                     fontsize=14, fontweight='bold', color='green')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 3, 3)
        ax3.imshow(real_B, cmap='gray', vmin=0, vmax=255)
        ax3.set_title(f'樣本{idx} - 專家標註\n(理想參考)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 第二行：局部放大（中心區域）
        h, w = real_A.shape
        cy, cx = h//2, w//2
        size = min(h, w) // 3
        crop_slice = (slice(cy-size//2, cy+size//2), slice(cx-size//2, cx+size//2))
        
        ax4 = plt.subplot(3, 3, 4)
        ax4.imshow(real_A[crop_slice], cmap='gray', vmin=0, vmax=255)
        ax4.set_title('原始 (放大)', fontsize=12)
        ax4.axis('off')
        
        ax5 = plt.subplot(3, 3, 5)
        ax5.imshow(fake_B[crop_slice], cmap='gray', vmin=0, vmax=255)
        ax5.set_title('增強 (放大)', fontsize=12, color='green')
        ax5.axis('off')
        
        ax6 = plt.subplot(3, 3, 6)
        ax6.imshow(real_B[crop_slice], cmap='gray', vmin=0, vmax=255)
        ax6.set_title('標註 (放大)', fontsize=12)
        ax6.axis('off')
        
        # 第三行：統計分析
        # 直方圖
        ax7 = plt.subplot(3, 3, 7)
        ax7.hist(real_A.ravel(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        ax7.set_title('原始像素分佈', fontsize=11)
        ax7.set_xlabel('像素值')
        ax7.set_ylabel('頻率')
        ax7.grid(True, alpha=0.3)
        # 添加統計信息
        stats_A = f'均值:{real_A.mean():.1f}\n標準差:{real_A.std():.1f}'
        ax7.text(0.98, 0.97, stats_A, transform=ax7.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax8 = plt.subplot(3, 3, 8)
        ax8.hist(fake_B.ravel(), bins=50, color='green', alpha=0.7, edgecolor='black')
        ax8.set_title('增強像素分佈', fontsize=11, color='green', fontweight='bold')
        ax8.set_xlabel('像素值')
        ax8.set_ylabel('頻率')
        ax8.grid(True, alpha=0.3)
        stats_B = f'均值:{fake_B.mean():.1f}\n標準差:{fake_B.std():.1f}'
        ax8.text(0.98, 0.97, stats_B, transform=ax8.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 對比度提升比較
        ax9 = plt.subplot(3, 3, 9)
        
        # 計算對比度指標
        contrast_A = real_A.std()
        contrast_B = fake_B.std()
        snr_A = real_A.mean() / (real_A.std() + 1e-8)
        snr_B = fake_B.mean() / (fake_B.std() + 1e-8)
        
        metrics = {
            '對比度\n(標準差)': [contrast_A, contrast_B],
            '動態範圍\n(max-min)': [real_A.max()-real_A.min(), fake_B.max()-fake_B.min()],
            '信噪比\n(mean/std)': [snr_A, snr_B]
        }
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (label, values) in enumerate(metrics.items()):
            ax9.bar(i - width/2, values[0], width, label='原始' if i==0 else '', 
                   color='blue', alpha=0.7)
            ax9.bar(i + width/2, values[1], width, label='增強' if i==0 else '', 
                   color='green', alpha=0.7)
            # 標註提升比例
            improvement = (values[1] - values[0]) / values[0] * 100
            ax9.text(i, max(values)*1.05, f'{improvement:+.1f}%', 
                    ha='center', fontsize=9, fontweight='bold')
        
        ax9.set_ylabel('值', fontsize=10)
        ax9.set_title('圖像質量指標對比', fontsize=11, fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics.keys(), fontsize=9)
        ax9.legend(fontsize=9)
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'OCTA圖像增強演示 - 樣本 {idx}\n'
                    f'實現：對比度提升、噪聲抑制、血管結構增強',
                    fontsize=16, fontweight='bold', color='darkgreen')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_path = os.path.join(output_dir, f'comparison_sample_{idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 已保存: {output_path}")
    
    # 創建總覽圖
    create_overview(test_results_dir, output_dir, sample_ids[:3])
    
    print(f"\n{'='*70}")
    print(f"✅ 演示圖像已生成到: {output_dir}/")
    print(f"{'='*70}")

def create_overview(test_results_dir, output_dir, sample_ids):
    """創建多樣本總覽圖"""
    
    n_samples = len(sample_ids)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_ids):
        real_A = np.array(Image.open(os.path.join(test_results_dir, f'{idx}_real_A.png')).convert('L'))
        fake_B = np.array(Image.open(os.path.join(test_results_dir, f'{idx}_fake_B.png')).convert('L'))
        real_B = np.array(Image.open(os.path.join(test_results_dir, f'{idx}_real_B.png')).convert('L'))
        
        axes[i, 0].imshow(real_A, cmap='gray')
        axes[i, 0].set_title(f'樣本{idx}: 原始', fontsize=12)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(fake_B, cmap='gray')
        axes[i, 1].set_title(f'樣本{idx}: 增強', fontsize=12, color='green', fontweight='bold')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(real_B, cmap='gray')
        axes[i, 2].set_title(f'樣本{idx}: 標註', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.suptitle('多樣本增強效果總覽', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    overview_path = os.path.join(output_dir, 'overview_multiple_samples.png')
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 已保存總覽圖: {overview_path}")

def print_enhancement_summary(test_results_dir):
    """打印增強效果統計摘要"""
    
    print("\n" + "="*70)
    print("增強效果統計摘要")
    print("="*70)
    
    # 收集所有樣本的統計信息
    image_files = [f for f in os.listdir(test_results_dir) if f.endswith('_real_A.png')]
    
    contrast_improvements = []
    snr_improvements = []
    
    for img_file in image_files:
        idx = img_file.replace('_real_A.png', '')
        
        real_A = np.array(Image.open(os.path.join(test_results_dir, f'{idx}_real_A.png')).convert('L'))
        fake_B = np.array(Image.open(os.path.join(test_results_dir, f'{idx}_fake_B.png')).convert('L'))
        
        # 對比度提升
        contrast_A = real_A.std()
        contrast_B = fake_B.std()
        contrast_improvement = (contrast_B - contrast_A) / contrast_A * 100
        contrast_improvements.append(contrast_improvement)
        
        # 信噪比
        snr_A = real_A.mean() / (real_A.std() + 1e-8)
        snr_B = fake_B.mean() / (fake_B.std() + 1e-8)
        snr_improvement = (snr_B - snr_A) / snr_A * 100
        snr_improvements.append(snr_improvement)
    
    print(f"\n測試樣本數量: {len(image_files)}")
    print(f"\n對比度提升:")
    print(f"  平均: {np.mean(contrast_improvements):+.2f}%")
    print(f"  範圍: [{np.min(contrast_improvements):+.2f}%, {np.max(contrast_improvements):+.2f}%]")
    
    print(f"\n信噪比變化:")
    print(f"  平均: {np.mean(snr_improvements):+.2f}%")
    print(f"  範圍: [{np.min(snr_improvements):+.2f}%, {np.max(snr_improvements):+.2f}%]")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    # 測試結果目錄
    test_results_dir = 'pytorch-CycleGAN-and-pix2pix/results/rose_svc_pix2pix/test_400/images'
    
    if not os.path.exists(test_results_dir):
        print(f"❌ 錯誤：找不到測試結果目錄: {test_results_dir}")
        sys.exit(1)
    
    # 創建演示圖像
    create_side_by_side_comparison(test_results_dir, output_dir='demo_results')
    
    # 打印統計摘要
    print_enhancement_summary(test_results_dir)
    
    print("\n✅ 演示完成！請查看 demo_results/ 目錄中的圖像")
    print("   - comparison_sample_XX.png: 詳細的單樣本對比")
    print("   - overview_multiple_samples.png: 多樣本總覽")


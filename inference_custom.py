#!/usr/bin/env python3
"""
客戶自定義圖像推理腳本
對根目錄下test文件夾中的圖像進行增強處理
"""
import os
import sys
import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 配置中文字體支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'PingFang SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加項目路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch-CycleGAN-and-pix2pix'))

from models import create_model
from options.test_options import TestOptions


def load_model():
    """加載訓練好的模型"""
    print("正在加載模型...")
    
    # 設置選項
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(project_root, 'pytorch-CycleGAN-and-pix2pix', 'checkpoints')
    
    old_argv = sys.argv
    sys.argv = [
        'test.py',
        '--dataroot', os.path.join(project_root, 'Dataset', 'ROSE'),
        '--checkpoints_dir', checkpoints_dir,
        '--name', 'rose_svc_pix2pix',
        '--model', 'pix2pix',
        '--netG', 'unet_256',
        '--direction', 'AtoB',
        '--dataset_mode', 'rose',
        '--norm', 'batch',
        '--input_nc', '1',
        '--output_nc', '1',
        '--no_dropout',
        '--epoch', '400',  # 使用最終模型
    ]
    
    opt = TestOptions().parse()
    sys.argv = old_argv
    
    # 手動設置一些選項
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False
    
    # 設置設備
    import torch
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.gpu_ids = []  # 使用CPU
    
    # 創建模型
    model = create_model(opt)
    model.setup(opt)
    
    if hasattr(model, 'eval'):
        model.eval()
    
    print("✅ 模型加載完成")
    return model, opt


def preprocess_image(image_path, target_size=256):
    """預處理輸入圖像"""
    # 讀取圖像
    if image_path.lower().endswith(('.tif', '.tiff')):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # 嘗試用PIL讀取
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"無法讀取圖像: {image_path}")
    
    # 確保是灰度圖
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 記錄原始尺寸
    original_size = img.shape[:2]
    
    # 歸一化到0-255
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # 調整大小
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 轉換為tensor，歸一化到[-1, 1]
    img_tensor = torch.from_numpy(img_resized).float()
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5  # 歸一化到[-1, 1]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel維度
    
    return img_tensor, img, original_size


def postprocess_output(output_tensor, original_size=None):
    """後處理輸出圖像"""
    # 從tensor轉換回numpy
    output = output_tensor.squeeze().cpu().detach().numpy()
    
    # 從[-1, 1]轉換回[0, 255]
    output = ((output + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    
    # 如果需要，調整回原始大小
    if original_size is not None:
        output = cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return output


def run_inference(model, input_tensor):
    """運行模型推理"""
    with torch.no_grad():
        # 設置輸入
        model.set_input({'A': input_tensor, 'B': input_tensor, 'A_paths': '', 'B_paths': ''})
        # 運行推理
        model.test()
        # 獲取輸出
        visuals = model.get_current_visuals()
        output = visuals['fake_B']
    return output


def create_comparison_figure(original, enhanced, save_path, filename):
    """創建對比圖"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始圖像
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'原始圖像\n{filename}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 增強圖像
    axes[1].imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('GAN增強結果', fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # 差異圖（顯示增強效果）
    # 計算對比度提升
    orig_contrast = original.std()
    enh_contrast = enhanced.std()
    contrast_change = (enh_contrast - orig_contrast) / (orig_contrast + 1e-8) * 100
    
    # 創建差異熱圖
    diff = np.abs(enhanced.astype(float) - cv2.resize(original, (enhanced.shape[1], enhanced.shape[0])).astype(float))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'變化區域熱圖\n對比度提升: {contrast_change:+.1f}%', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'OCTA圖像增強效果 - {filename}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return contrast_change


def main():
    """主函數"""
    print("="*70)
    print("客戶測試圖像推理")
    print("="*70)
    
    # 設置路徑
    input_dir = os.path.join(os.path.dirname(__file__), 'test')
    output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    
    # 檢查輸入目錄
    if not os.path.exists(input_dir):
        print(f"❌ 錯誤：找不到測試目錄 {input_dir}")
        return
    
    # 獲取所有圖像文件
    valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"❌ 錯誤：測試目錄中沒有找到圖像文件")
        return
    
    print(f"\n找到 {len(image_files)} 個測試圖像:")
    for f in image_files:
        print(f"  - {f}")
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 加載模型
    model, opt = load_model()
    
    # 處理每個圖像
    print("\n" + "="*70)
    print("開始處理圖像...")
    print("="*70)
    
    results = []
    
    for filename in image_files:
        print(f"\n處理: {filename}")
        
        input_path = os.path.join(input_dir, filename)
        
        try:
            # 預處理
            input_tensor, original_img, original_size = preprocess_image(input_path)
            print(f"  原始尺寸: {original_size}")
            
            # 推理
            output_tensor = run_inference(model, input_tensor)
            
            # 後處理（恢復原始尺寸）
            enhanced_img = postprocess_output(output_tensor, original_size)
            
            # 保存增強後的圖像
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
            Image.fromarray(enhanced_img).save(output_path)
            print(f"  ✅ 增強圖像已保存: {output_path}")
            
            # 創建對比圖
            comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            contrast_change = create_comparison_figure(original_img, enhanced_img, comparison_path, filename)
            print(f"  ✅ 對比圖已保存: {comparison_path}")
            print(f"  📊 對比度提升: {contrast_change:+.1f}%")
            
            results.append({
                'filename': filename,
                'contrast_change': contrast_change,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  ❌ 處理失敗: {str(e)}")
            results.append({
                'filename': filename,
                'status': 'failed',
                'error': str(e)
            })
    
    # 創建總覽圖
    print("\n" + "="*70)
    print("生成總覽圖...")
    print("="*70)
    
    create_overview(output_dir, image_files)
    
    # 打印摘要
    print("\n" + "="*70)
    print("處理完成摘要")
    print("="*70)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"\n成功: {len(successful)}/{len(results)}")
    if successful:
        contrasts = [r['contrast_change'] for r in successful]
        print(f"對比度提升:")
        print(f"  平均: {np.mean(contrasts):+.1f}%")
        print(f"  範圍: [{min(contrasts):+.1f}%, {max(contrasts):+.1f}%]")
    
    if failed:
        print(f"\n失敗: {len(failed)}")
        for r in failed:
            print(f"  - {r['filename']}: {r.get('error', 'Unknown error')}")
    
    print(f"\n✅ 所有結果已保存到: {output_dir}/")
    print("  - *_enhanced.png: 增強後的圖像")
    print("  - *_comparison.png: 對比圖")
    print("  - overview.png: 總覽圖")


def create_overview(output_dir, image_files):
    """創建所有測試圖像的總覽圖"""
    n_images = len(image_files)
    if n_images == 0:
        return
    
    # 計算網格大小
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols * 2 == 2:
        axes = np.array([[axes[0], axes[1]]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, filename in enumerate(image_files):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        
        base_name = os.path.splitext(filename)[0]
        
        # 讀取原始圖像
        input_path = os.path.join(os.path.dirname(__file__), 'test', filename)
        try:
            if filename.lower().endswith(('.tif', '.tiff')):
                orig_img = np.array(Image.open(input_path))
            else:
                orig_img = np.array(Image.open(input_path).convert('L'))
            
            if len(orig_img.shape) == 3:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        except:
            continue
        
        # 讀取增強圖像
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
        try:
            enh_img = np.array(Image.open(enhanced_path).convert('L'))
        except:
            continue
        
        # 顯示原始圖像
        axes[row, col].imshow(orig_img, cmap='gray')
        axes[row, col].set_title(f'{base_name}\n原始', fontsize=10)
        axes[row, col].axis('off')
        
        # 顯示增強圖像
        axes[row, col + 1].imshow(enh_img, cmap='gray')
        axes[row, col + 1].set_title(f'{base_name}\n增強', fontsize=10, color='green')
        axes[row, col + 1].axis('off')
    
    # 隱藏空白子圖
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')
    
    plt.suptitle('客戶測試圖像增強效果總覽', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 總覽圖已保存: {os.path.join(output_dir, 'overview.png')}")


if __name__ == '__main__':
    main()


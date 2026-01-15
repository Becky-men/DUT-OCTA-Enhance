#!/usr/bin/env python3
"""
交互式推理脚本 - 客户自定义图像增强

这个脚本提供交互式菜单，让用户选择：
1. 使用哪个已训练的模型
2. 选择输入文件或文件夹（支持test文件夹或自定义路径）
3. 选择输出目录
4. 自动处理图像并生成对比结果
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
from pathlib import Path

# 配置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'PingFang SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pytorch-CycleGAN-and-pix2pix'))

from models import create_model
from options.test_options import TestOptions
from util.util import init_ddp

# 颜色输出
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """打印标题"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")

def print_option(num, text, color=Colors.OKCYAN):
    """打印选项"""
    print(f"{color}[{num}]{Colors.ENDC} {text}")

def print_success(text):
    """打印成功信息"""
    print(f"{Colors.OKGREEN}[成功] {text}{Colors.ENDC}")

def print_warning(text):
    """打印警告信息"""
    print(f"{Colors.WARNING}[警告] {text}{Colors.ENDC}")

def print_error(text):
    """打印错误信息"""
    print(f"{Colors.FAIL}[错误] {text}{Colors.ENDC}")

def print_info(text):
    """打印信息"""
    print(f"{Colors.OKBLUE}[信息] {text}{Colors.ENDC}")


def list_available_models(checkpoints_dir):
    """列出可用的训练模型"""
    models = []
    if not os.path.exists(checkpoints_dir):
        return models
    
    for item in os.listdir(checkpoints_dir):
        model_path = os.path.join(checkpoints_dir, item)
        if os.path.isdir(model_path):
            # 检查是否有模型文件
            latest_g = os.path.join(model_path, 'latest_net_G.pth')
            if os.path.exists(latest_g):
                models.append(item)
            else:
                # 检查是否有epoch模型文件
                epoch_files = [f for f in os.listdir(model_path) if f.endswith('_net_G.pth')]
                if epoch_files:
                    models.append(item)
    
    return sorted(models)


def select_model(checkpoints_dir):
    """交互式选择模型"""
    print_header("选择训练好的模型")
    
    models = list_available_models(checkpoints_dir)
    
    if not models:
        print_error(f"在 {checkpoints_dir} 中没有找到训练好的模型！")
        print_info("请先训练模型，或检查checkpoints目录路径。")
        return None
    
    print(f"\n找到 {len(models)} 个可用模型：\n")
    for idx, model in enumerate(models, 1):
        model_path = os.path.join(checkpoints_dir, model)
        # 检查模型信息
        latest_g = os.path.join(model_path, 'latest_net_G.pth')
        if os.path.exists(latest_g):
            print_option(idx, f"{model} (最新模型)")
        else:
            epoch_files = [f for f in os.listdir(model_path) if f.endswith('_net_G.pth')]
            if epoch_files:
                print_option(idx, f"{model} ({len(epoch_files)} 个检查点)")
    
    print_option(0, "退出")
    
    while True:
        try:
            choice = input(f"\n{Colors.OKCYAN}请选择模型 [1-{len(models)}] 或 0 退出: {Colors.ENDC}").strip()
            choice = int(choice)
            
            if choice == 0:
                return None
            elif 1 <= choice <= len(models):
                selected_model = models[choice - 1]
                print_success(f"已选择模型: {selected_model}")
                return selected_model
            else:
                print_error(f"无效选择，请输入 1-{len(models)} 或 0")
        except ValueError:
            print_error("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消")
            return None


def select_input_source():
    """交互式选择输入源"""
    print_header("选择输入图像源")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    default_test_dir = os.path.join(project_root, 'test')
    current_dir = os.getcwd()
    
    print_option(1, f"使用默认test文件夹")
    print(f"     路径: {default_test_dir}")
    print_option(2, "指定自定义文件夹（支持拖拽文件夹）")
    print_option(3, "指定单个图像文件（支持拖拽文件）")
    print_option(0, "退出")
    print(f"\n{Colors.OKBLUE}提示: 当前工作目录: {current_dir}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}提示: 可以使用相对路径（相对于当前目录）或绝对路径{Colors.ENDC}")
    print(f"{Colors.OKBLUE}提示: macOS用户可以拖拽文件夹/文件到终端，路径会自动填入{Colors.ENDC}")
    
    while True:
        try:
            choice = input(f"\n{Colors.OKCYAN}请选择 [1-3] 或 0 退出: {Colors.ENDC}").strip()
            choice = int(choice)
            
            if choice == 0:
                return None, None
            elif choice == 1:
                if os.path.exists(default_test_dir):
                    print_success(f"使用默认test文件夹: {default_test_dir}")
                    return default_test_dir, 'folder'
                else:
                    print_error(f"默认test文件夹不存在: {default_test_dir}")
                    print_warning("请选择其他选项")
                    continue
            elif choice == 2:
                while True:
                    folder_path = input(f"\n{Colors.OKCYAN}请输入文件夹路径（或拖拽文件夹）: {Colors.ENDC}").strip()
                    
                    # 移除可能的引号（拖拽文件时可能带引号）
                    folder_path = folder_path.strip('"').strip("'")
                    
                    # 展开用户目录和相对路径
                    folder_path = os.path.expanduser(folder_path)
                    if not os.path.isabs(folder_path):
                        # 如果是相对路径，相对于当前工作目录
                        folder_path = os.path.join(current_dir, folder_path)
                    
                    folder_path = os.path.abspath(folder_path)
                    
                    if os.path.isdir(folder_path):
                        # 检查文件夹中是否有图像文件
                        valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
                        image_files = [f for f in os.listdir(folder_path) 
                                     if f.lower().endswith(valid_extensions)]
                        if image_files:
                            print_success(f"已选择文件夹: {folder_path}")
                            print_info(f"找到 {len(image_files)} 个图像文件")
                            return folder_path, 'folder'
                        else:
                            print_warning(f"文件夹中没有找到图像文件（支持: {', '.join(valid_extensions)}）")
                            retry = input(f"{Colors.OKCYAN}是否重新输入? [y/N]: {Colors.ENDC}").strip().lower()
                            if retry != 'y':
                                break
                    else:
                        print_error(f"文件夹不存在: {folder_path}")
                        retry = input(f"{Colors.OKCYAN}是否重新输入? [y/N]: {Colors.ENDC}").strip().lower()
                        if retry != 'y':
                            break
            elif choice == 3:
                while True:
                    file_path = input(f"\n{Colors.OKCYAN}请输入图像文件路径（或拖拽文件）: {Colors.ENDC}").strip()
                    
                    # 移除可能的引号（拖拽文件时可能带引号）
                    file_path = file_path.strip('"').strip("'")
                    
                    # 展开用户目录和相对路径
                    file_path = os.path.expanduser(file_path)
                    if not os.path.isabs(file_path):
                        # 如果是相对路径，相对于当前工作目录
                        file_path = os.path.join(current_dir, file_path)
                    
                    file_path = os.path.abspath(file_path)
                    
                    if os.path.isfile(file_path):
                        valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
                        if file_path.lower().endswith(valid_extensions):
                            print_success(f"已选择文件: {file_path}")
                            return file_path, 'file'
                        else:
                            print_error(f"不支持的文件格式（支持: {', '.join(valid_extensions)}）")
                            retry = input(f"{Colors.OKCYAN}是否重新输入? [y/N]: {Colors.ENDC}").strip().lower()
                            if retry != 'y':
                                break
                    else:
                        print_error(f"文件不存在: {file_path}")
                        retry = input(f"{Colors.OKCYAN}是否重新输入? [y/N]: {Colors.ENDC}").strip().lower()
                        if retry != 'y':
                            break
            else:
                print_error("无效选择，请输入 1-3 或 0")
        except ValueError:
            print_error("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消")
            return None, None


def select_output_dir(default_output_dir):
    """交互式选择输出目录"""
    print_header("选择输出目录")
    
    current_dir = os.getcwd()
    
    print_option(1, f"使用默认输出目录")
    print(f"     路径: {default_output_dir}")
    print_option(2, "指定自定义输出目录（支持拖拽文件夹）")
    print_option(0, "退出")
    print(f"\n{Colors.OKBLUE}提示: 当前工作目录: {current_dir}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}提示: 可以使用相对路径（相对于当前目录）或绝对路径{Colors.ENDC}")
    print(f"{Colors.OKBLUE}提示: macOS用户可以拖拽文件夹到终端，路径会自动填入{Colors.ENDC}")
    
    while True:
        try:
            choice = input(f"\n{Colors.OKCYAN}请选择 [1-2] 或 0 退出: {Colors.ENDC}").strip()
            choice = int(choice)
            
            if choice == 0:
                return None
            elif choice == 1:
                os.makedirs(default_output_dir, exist_ok=True)
                print_success(f"使用默认输出目录: {default_output_dir}")
                return default_output_dir
            elif choice == 2:
                while True:
                    output_dir = input(f"\n{Colors.OKCYAN}请输入输出目录路径（或拖拽文件夹）: {Colors.ENDC}").strip()
                    
                    # 移除可能的引号（拖拽文件时可能带引号）
                    output_dir = output_dir.strip('"').strip("'")
                    
                    # 展开用户目录和相对路径
                    output_dir = os.path.expanduser(output_dir)
                    if not os.path.isabs(output_dir):
                        # 如果是相对路径，相对于当前工作目录
                        output_dir = os.path.join(current_dir, output_dir)
                    
                    output_dir = os.path.abspath(output_dir)
                    
                    try:
                        os.makedirs(output_dir, exist_ok=True)
                        print_success(f"已选择输出目录: {output_dir}")
                        return output_dir
                    except Exception as e:
                        print_error(f"无法创建输出目录: {str(e)}")
                        retry = input(f"{Colors.OKCYAN}是否重新输入? [y/N]: {Colors.ENDC}").strip().lower()
                        if retry != 'y':
                            break
            else:
                print_error("无效选择，请输入 1-2 或 0")
        except ValueError:
            print_error("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n已取消")
            return None


def get_image_files(input_source, source_type):
    """获取要处理的图像文件列表"""
    valid_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')
    
    if source_type == 'file':
        if input_source.lower().endswith(valid_extensions):
            return [input_source]
        else:
            print_error(f"不支持的文件格式: {input_source}")
            return []
    else:  # folder
        image_files = []
        for f in os.listdir(input_source):
            if f.lower().endswith(valid_extensions):
                image_files.append(os.path.join(input_source, f))
        return sorted(image_files)


def load_model(model_name, checkpoints_dir, project_root):
    """加载训练好的模型"""
    print_info("正在加载模型...")
    
    old_argv = sys.argv
    sys.argv = [
        'test.py',
        '--dataroot', os.path.join(project_root, 'Dataset', 'ROSE'),
        '--checkpoints_dir', checkpoints_dir,
        '--name', model_name,
        '--model', 'pix2pix',
        '--netG', 'unet_256',
        '--direction', 'AtoB',
        '--dataset_mode', 'rose',
        '--norm', 'batch',
        '--input_nc', '1',
        '--output_nc', '1',
        '--no_dropout',
        '--epoch', 'latest',  # 使用最新模型
    ]
    
    opt = TestOptions().parse()
    sys.argv = old_argv
    
    # 手动设置一些选项
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.isTrain = False
    
    # 自动选择设备
    opt.device = init_ddp()
    opt.gpu_ids = []
    
    # 创建模型
    model = create_model(opt)
    model.setup(opt)
    
    if hasattr(model, 'eval'):
        model.eval()
    
    print_success("模型加载完成")
    print_info(f"使用设备: {opt.device}")
    return model, opt


def preprocess_image(image_path, target_size=256):
    """预处理输入图像"""
    # 读取图像
    if image_path.lower().endswith(('.tif', '.tiff')):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # 尝试用PIL读取
            pil_img = Image.open(image_path)
            img = np.array(pil_img)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 确保是灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 记录原始尺寸
    original_size = img.shape[:2]
    
    # 归一化到0-255
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    
    # 调整大小
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 转换为tensor，归一化到[-1, 1]
    img_tensor = torch.from_numpy(img_resized).float()
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5  # 归一化到[-1, 1]
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度
    
    return img_tensor, img, original_size


def postprocess_output(output_tensor, original_size=None):
    """后处理输出图像"""
    # 从tensor转换回numpy
    output = output_tensor.squeeze().cpu().detach().numpy()
    
    # 从[-1, 1]转换回[0, 255]
    output = ((output + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    
    # 如果需要，调整回原始大小
    if original_size is not None:
        output = cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
    
    return output


def run_inference(model, input_tensor, device):
    """运行模型推理"""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        # 设置输入
        model.set_input({'A': input_tensor, 'B': input_tensor, 'A_paths': '', 'B_paths': ''})
        # 运行推理
        model.test()
        # 获取输出
        visuals = model.get_current_visuals()
        output = visuals['fake_B']
    return output


def create_comparison_figure(original, enhanced, save_path, filename):
    """创建对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原始图像
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f'原始图像\n{filename}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 增强图像
    axes[1].imshow(enhanced, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('GAN增强结果', fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # 差异图（显示增强效果）
    orig_resized = cv2.resize(original, (enhanced.shape[1], enhanced.shape[0]))
    diff = np.abs(enhanced.astype(float) - orig_resized.astype(float))
    axes[2].imshow(diff, cmap='hot')
    
    # 计算对比度提升
    orig_contrast = original.std()
    enh_contrast = enhanced.std()
    contrast_change = (enh_contrast - orig_contrast) / (orig_contrast + 1e-8) * 100
    
    axes[2].set_title(f'变化区域热图\n对比度提升: {contrast_change:+.1f}%', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f'OCTA图像增强效果 - {filename}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return contrast_change


def create_overview(output_dir, image_files, input_source, source_type):
    """创建所有测试图像的总览图"""
    n_images = len(image_files)
    if n_images == 0:
        return
    
    # 计算网格大小
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols * 2 == 2:
        axes = np.array([[axes[0], axes[1]]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, image_path in enumerate(image_files):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        
        # 读取原始图像
        try:
            if filename.lower().endswith(('.tif', '.tiff')):
                orig_img = np.array(Image.open(image_path))
            else:
                orig_img = np.array(Image.open(image_path).convert('L'))
            
            if len(orig_img.shape) == 3:
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
        except:
            continue
        
        # 读取增强图像
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
        try:
            enh_img = np.array(Image.open(enhanced_path).convert('L'))
        except:
            continue
        
        # 显示原始图像
        axes[row, col].imshow(orig_img, cmap='gray')
        axes[row, col].set_title(f'{base_name}\n原始', fontsize=10)
        axes[row, col].axis('off')
        
        # 显示增强图像
        axes[row, col + 1].imshow(enh_img, cmap='gray')
        axes[row, col + 1].set_title(f'{base_name}\n增强', fontsize=10, color='green')
        axes[row, col + 1].axis('off')
    
    # 隐藏空白子图
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = (idx % n_cols) * 2
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis('off')
            if col + 1 < axes.shape[1]:
                axes[row, col + 1].axis('off')
    
    plt.suptitle('图像增强效果总览', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print_success(f"总览图已保存: {os.path.join(output_dir, 'overview.png')}")


def process_images(model, opt, image_files, output_dir):
    """处理所有图像"""
    print_header("开始处理图像")
    
    results = []
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        print(f"\n处理: {filename}")
        
        try:
            # 预处理
            input_tensor, original_img, original_size = preprocess_image(image_path)
            print_info(f"原始尺寸: {original_size}")
            
            # 推理
            output_tensor = run_inference(model, input_tensor, opt.device)
            
            # 后处理（恢复原始尺寸）
            enhanced_img = postprocess_output(output_tensor, original_size)
            
            # 保存增强后的图像
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
            Image.fromarray(enhanced_img).save(output_path)
            print_success(f"增强图像已保存: {output_path}")
            
            # 创建对比图
            comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            contrast_change = create_comparison_figure(original_img, enhanced_img, comparison_path, filename)
            print_success(f"对比图已保存: {comparison_path}")
            print_info(f"对比度提升: {contrast_change:+.1f}%")
            
            results.append({
                'filename': filename,
                'contrast_change': contrast_change,
                'status': 'success'
            })
            
        except Exception as e:
            print_error(f"处理失败: {str(e)}")
            results.append({
                'filename': filename,
                'status': 'failed',
                'error': str(e)
            })
    
    return results


def main():
    """主函数"""
    print_header("ROSE数据集GAN图像增强 - 交互式推理")
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(project_root, 'pytorch-CycleGAN-and-pix2pix', 'checkpoints')
    default_output_dir = os.path.join(project_root, 'test_results')
    
    try:
        # 1. 选择模型
        model_name = select_model(checkpoints_dir)
        if model_name is None:
            print_warning("已取消操作")
            return
        
        # 2. 选择输入源
        input_source, source_type = select_input_source()
        if input_source is None:
            print_warning("已取消操作")
            return
        
        # 3. 选择输出目录
        output_dir = select_output_dir(default_output_dir)
        if output_dir is None:
            print_warning("已取消操作")
            return
        
        # 4. 获取图像文件列表
        image_files = get_image_files(input_source, source_type)
        if not image_files:
            print_error("没有找到可处理的图像文件")
            return
        
        print_header("配置摘要")
        print(f"模型: {model_name}")
        print(f"输入源: {input_source} ({source_type})")
        print(f"输出目录: {output_dir}")
        print(f"图像数量: {len(image_files)}")
        
        confirm = input(f"\n{Colors.OKCYAN}确认开始处理? [y/N]: {Colors.ENDC}").strip().lower()
        if confirm != 'y':
            print_warning("已取消操作")
            return
        
        # 5. 加载模型
        model, opt = load_model(model_name, checkpoints_dir, project_root)
        
        # 6. 处理图像
        results = process_images(model, opt, image_files, output_dir)
        
        # 7. 创建总览图
        if len(image_files) > 1:
            print_header("生成总览图")
            create_overview(output_dir, image_files, input_source, source_type)
        
        # 8. 打印摘要
        print_header("处理完成摘要")
        
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']
        
        print(f"\n成功: {len(successful)}/{len(results)}")
        if successful:
            contrasts = [r['contrast_change'] for r in successful]
            print(f"对比度提升:")
            print(f"  平均: {np.mean(contrasts):+.1f}%")
            print(f"  范围: [{min(contrasts):+.1f}%, {max(contrasts):+.1f}%]")
        
        if failed:
            print(f"\n失败: {len(failed)}")
            for r in failed:
                print(f"  - {r['filename']}: {r.get('error', 'Unknown error')}")
        
        print(f"\n{Colors.OKGREEN}所有结果已保存到: {output_dir}/{Colors.ENDC}")
        print("  - *_enhanced.png: 增强后的图像")
        print("  - *_comparison.png: 对比图")
        if len(image_files) > 1:
            print("  - overview.png: 总览图")
        
    except KeyboardInterrupt:
        print("\n\n已取消操作")
    except Exception as e:
        print_error(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


# ROSE数据集GAN图像增强项目

## 项目概述

OCTA图像增强系统，支持ROSE-1和ROSE-2数据集。ROSE-1支持SVC、DVC、SVC_DVC三种血管层模式，ROSE-2支持SVP模式。训练和推理时会自动选择计算设备（CUDA/MPS/CPU）。

## 技术架构

### 核心模型

使用条件生成对抗网络（Conditional GAN）架构，包含生成器和判别器两个核心组件。

#### 生成器（Generator）- U-Net架构

生成器使用U-Net编码器-解码器结构，将低质量OCTA图像转换为增强后的高质量图像。

**架构特点：**
1. **输入尺寸**: 256×256 灰度图像（1通道）
2. **输出尺寸**: 256×256 灰度图像（1通道）
3. **网络结构**: 8层下采样 + 8层上采样，带跳跃连接（Skip Connections）
4. **下采样路径（编码器）**: 
   1. 逐步降低特征图尺寸，提取图像特征
   2. 使用卷积层和批量归一化（Batch Normalization）
   3. 通道数逐步增加：1 → 64 → 128 → 256 → 512 → 512 → 512 → 512 → 512
5. **上采样路径（解码器）**: 
   1. 逐步恢复图像尺寸，生成增强图像
   2. 使用转置卷积（Transposed Convolution）和批量归一化
   3. 通道数逐步减少：512 → 512 → 512 → 512 → 256 → 128 → 64 → 1
6. **跳跃连接**: 将编码器特征直接连接到解码器对应层，保留细节信息

**架构示意图：**
```
输入图像 (256×256×1)
    ↓
[编码器 - 下采样]
    ↓
256×256×64 → 128×128×128 → 64×64×256 → 32×32×512
    ↓           ↓             ↓            ↓
[瓶颈层] → 16×16×512 → 8×8×512 → 4×4×512 → 2×2×512 → 1×1×512
    ↓
[解码器 - 上采样 + 跳跃连接]
    ↓
1×1×512 → 2×2×512 → 4×4×512 → 8×8×512 → 16×16×512
    ↓           ↓             ↓            ↓
32×32×256 → 64×64×128 → 128×128×64 → 256×256×1
    ↓
输出图像 (256×256×1)
```

#### 判别器（Discriminator）- PatchGAN架构

判别器使用PatchGAN架构，对图像的局部区域（70×70像素块）进行真假判断，而非整张图像。

**架构特点：**
1. **输入**: 拼接的输入图像和输出图像（2通道：原始图像 + 生成图像）
2. **输出**: 30×30 的真假判断图（每个像素对应输入图像的一个70×70区域）
3. **网络结构**: 3层卷积 + 1层输出卷积
4. **卷积层配置**:
   1. 第1层: 64个滤波器，步长2，输出128×128×64
   2. 第2层: 128个滤波器，步长2，输出64×64×128
   3. 第3层: 256个滤波器，步长2，输出32×32×256
   4. 输出层: 1个滤波器，步长1，输出30×30×1
5. **激活函数**: LeakyReLU（负斜率为0.2）
6. **归一化**: 批量归一化（Batch Normalization）

**架构示意图：**
```
输入: [原始图像, 生成图像] (256×256×2)
    ↓
[卷积层1] 64 filters, stride=2
    ↓
128×128×64
    ↓
[卷积层2] 128 filters, stride=2
    ↓
64×64×128
    ↓
[卷积层3] 256 filters, stride=2
    ↓
32×32×256
    ↓
[输出层] 1 filter, stride=1
    ↓
输出: 30×30×1 (每个值表示对应70×70区域为真的概率)
```

#### 损失函数

训练过程使用多组件损失函数，使生成图像既真实又准确。

**1. GAN损失（对抗损失）**
1. **目的**: 让生成图像能够欺骗判别器，使其看起来真实
2. **公式**: `L_GAN = log(D(x, G(x)))`，其中G是生成器，D是判别器
3. **作用**: 使生成图像具有真实感

**2. L1损失（重建损失）**
1. **目的**: 使生成图像与标注图像在像素级别上接近
2. **公式**: `L_L1 = ||G(x) - y||_1`，其中y是真实标注图像
3. **权重**: λ_L1 = 100.0（默认值）
4. **作用**: 保持生成图像的准确性，避免过度模糊

**3. 结构约束损失（仅ROSE-1）**

**thin_gt损失（骨架化血管约束）:**
1. **目的**: 约束生成图像在血管骨架区域与标注图像一致
2. **方法**: 使用thin_gt作为掩膜，只计算血管骨架区域的L1损失
3. **公式**: `L_thin = ||(G(x) - y) ⊙ M_thin||_1`，其中M_thin是thin_gt掩膜
4. **权重**: λ_thin = 10.0（默认值）
5. **作用**: 使血管骨架结构准确

**thick_gt损失（完整血管区域约束）:**
1. **目的**: 约束生成图像在完整血管区域与标注图像一致
2. **方法**: 使用thick_gt作为掩膜，只计算完整血管区域的L1损失
3. **公式**: `L_thick = ||(G(x) - y) ⊙ M_thick||_1`，其中M_thick是thick_gt掩膜
4. **权重**: λ_thick = 10.0（默认值）
5. **作用**: 使完整血管区域准确

**总损失函数：**
```
L_total = L_GAN + λ_L1 × L_L1 + λ_thin × L_thin + λ_thick × L_thick
```

**损失函数示意图：**
```
输入图像 (real_A)
    ↓
生成器 (G)
    ↓
生成图像 (fake_B)
    ↓
    ├─→ [与real_B比较] → L1损失
    │
    ├─→ [与thin_gt掩膜区域比较] → thin_gt损失（可选）
    │
    ├─→ [与thick_gt掩膜区域比较] → thick_gt损失（可选）
    │
    └─→ [与real_A拼接] → 判别器 (D) → GAN损失
```

#### 训练流程

训练过程采用交替优化策略，每个批次按顺序更新判别器和生成器。

**每个训练批次的执行顺序：**

1. **数据加载**: 从数据集中加载一个批次，包含输入图像（real_A）、标注图像（real_B），以及可选的thin_gt和thick_gt

2. **生成器前向传播**: 输入图像通过生成器网络得到生成图像 `fake_B = G(real_A)`

3. **更新判别器**（先执行）:
   1. 拼接输入图像和生成图像，输入判别器计算假图像对的损失（应判断为假）
   2. 拼接输入图像和真实标注图像，输入判别器计算真图像对的损失（应判断为真）
   3. 合并两个损失，反向传播更新判别器参数
   4. 注意：计算假图像损失时使用`detach()`停止对生成器的梯度传播

4. **更新生成器**（后执行）:
   1. 拼接输入图像和生成图像，输入判别器计算GAN损失（生成图像应欺骗判别器）
   2. 计算L1损失（生成图像与真实标注的像素级差异）
   3. 如果使用thin_gt/thick_gt，计算结构约束损失（在血管区域应用掩膜后的L1损失）
   4. 合并所有损失，反向传播更新生成器参数

**训练循环结构**（基于`train.py`）:
1. 外层循环：遍历所有epoch（默认200个epoch + 200个衰减epoch）
2. 内层循环：遍历每个epoch中的所有数据批次
3. 每个批次：调用`model.set_input(data)`设置输入，然后调用`model.optimize_parameters()`执行上述步骤3和4

### 数据集支持

#### ROSE-1数据集

支持三种血管层模式：
1. **SVC** (Superficial Vascular Complex): 浅层血管复合体
2. **DVC** (Deep Vascular Complex): 深层血管复合体  
3. **SVC_DVC**: 融合层

目录结构：
```
ROSE-1/
    {模式}/
        train/
            img/      输入图像（待增强）
            gt/       标注图像（理想血管结构）
            thin_gt/  骨架化血管标注（可选）
            thick_gt/ 完整血管掩膜（可选）
        test/
            结构同上
```

#### ROSE-2数据集

支持SVP（Superficial Vascular Plexus）模式：

目录结构：
```
ROSE-2/
    train/
        original/  原始输入图像
        gt/        标注图像
    test/
        结构同上
```

### 数据预处理流程

图像在输入网络前会经过标准化预处理流程：

**预处理步骤：**
1. **格式转换**: 自动识别PNG和TIF格式，统一转换为灰度图像
2. **尺寸调整**: 加载时调整为286×286，然后随机裁剪为256×256
3. **归一化**: 将像素值从[0, 255]归一化到[-1, 1]范围
4. **数据增强**（训练时）:
   1. 随机水平翻转（50%概率）
   2. 随机裁剪位置

**数据流处理顺序**:
原始图像（任意尺寸，像素值范围[0,255]） → 格式转换（PNG/TIF自动识别，统一转为灰度图） → 尺寸调整（resize到286×286） → 随机裁剪（训练时随机位置裁剪到256×256，测试时中心裁剪） → 归一化（像素值从[0,255]映射到[-1,1]） → 转换为张量格式（1×256×256） → 输入网络

### 推理流程

推理过程将训练好的模型应用于新的输入图像，生成增强结果。

**推理步骤：**

1. **图像预处理**:
   1. 读取输入图像（支持PNG、TIF格式）
   2. 转换为灰度图像
   3. 调整尺寸到256×256
   4. 归一化到[-1, 1]范围

2. **模型推理**:
   1. 加载训练好的生成器权重
   2. 输入图像通过生成器网络
   3. 生成增强后的图像

3. **后处理**:
   1. 将输出从[-1, 1]范围转换回[0, 255]
   2. 恢复原始图像尺寸（如果需要）
   3. 保存为PNG格式

**推理处理顺序**:
输入图像（任意尺寸，PNG/TIF格式） → 预处理（格式转换到灰度图，resize到256×256，归一化到[-1,1]） → 转换为张量格式（1×256×256） → 生成器网络推理（编码器提取特征，解码器生成图像，跳跃连接保留细节） → 生成图像张量（1×256×256，值域[-1,1]） → 后处理（反归一化到[0,255]，resize回原始尺寸，保存为PNG格式） → 输出图像（原始尺寸，PNG格式）

### 设备自动选择

训练和推理时会自动检测并选择计算设备，优先级如下：
1. CUDA（NVIDIA GPU，如果可用）
2. MPS（Apple Silicon GPU，如果可用）
3. CPU（后备选项）

**设备选择逻辑**（基于`util/util.py`的`init_ddp()`函数）:
启动训练/推理时，系统按以下顺序自动检测设备：首先检查环境变量`WORLD_SIZE`，如果存在且大于1则进入DDP模式并使用CUDA设备；否则检查`torch.cuda.is_available()`，如果为True则使用CUDA设备；否则检查`torch.backends.mps.is_available()`，如果为True则使用MPS设备（Apple Silicon GPU）；最后如果以上都不可用，则使用CPU设备。

## 技术细节

### 网络架构详细说明

#### U-Net生成器详细结构

U-Net生成器采用对称的编码器-解码器结构，通过跳跃连接保留细节信息。

**编码器（下采样）部分：**
1. 每层使用卷积+批量归一化+ReLU激活
2. 步长为2的卷积实现下采样
3. 通道数逐步增加：1 → 64 → 128 → 256 → 512

**瓶颈层：**
1. 最深层特征提取，尺寸为1×1或更小
2. 包含多个残差块（如果使用ResNet架构）

**解码器（上采样）部分：**
1. 每层使用转置卷积+批量归一化+ReLU激活
2. 步长为2的转置卷积实现上采样
3. 通道数逐步减少：512 → 256 → 128 → 64 → 1
4. 通过跳跃连接融合编码器对应层的特征

**关键特性：**
1. **跳跃连接**: 将编码器第i层的特征直接连接到解码器第(n-i)层，保留细节
2. **批量归一化**: 加速训练，提高稳定性
3. **Dropout**: 在训练时随机丢弃部分连接，防止过拟合（仅在解码器中间层使用）

#### PatchGAN判别器工作原理

PatchGAN不是对整张图像进行真假判断，而是对图像的局部区域（patch）进行判断。

**工作原理：**
1. 输入：256×256的图像对（原始图像+生成图像，2通道）
2. 输出：30×30的判断图，每个值对应输入图像中一个70×70区域的真假概率
3. 优势：
   1. 参数量少，训练快
   2. 可以处理任意尺寸的图像（全卷积结构）
   3. 关注局部纹理和细节，适合图像到图像转换任务

**判断过程**（基于`networks.py`的`NLayerDiscriminator`）:
输入图像对（256×256×2通道） → 卷积层1（4×4卷积，步长2，输出128×128×64） → 卷积层2（4×4卷积，步长2，输出64×64×128） → 卷积层3（4×4卷积，步长2，输出32×32×256） → 输出层（4×4卷积，步长1，输出30×30×1） → 每个输出值对应输入图像中一个70×70区域的真假概率

### 结构约束损失详解

ROSE-1数据集提供了额外的结构标注信息，用于约束生成图像的血管结构。

**thin_gt（骨架化血管）:**
1. **含义**: 血管的骨架化表示，只保留血管的中心线
2. **用途**: 约束生成图像在血管中心线位置与标注图像一致
3. **计算方式**: 
   ```python
   thin_mask = (thin_gt + 1.0) / 2.0  # 归一化到[0,1]
   fake_masked = fake_B * thin_mask    # 应用掩膜
   real_masked = real_B * thin_mask
   loss = L1(fake_masked, real_masked) # 计算掩膜区域的L1损失
   ```

**thick_gt（完整血管掩膜）:**
1. **含义**: 完整血管区域的二值掩膜
2. **用途**: 约束生成图像在整个血管区域与标注图像一致
3. **计算方式**: 与thin_gt类似，但掩膜覆盖范围更大

**结构约束计算过程**（基于`pix2pix_model.py`的`backward_G()`方法）:
对于thin_gt损失：生成图像（fake_B） → 将thin_gt从[-1,1]归一化到[0,1]作为掩膜 → 应用掩膜提取血管骨架区域 → 与标注图像的骨架区域计算L1损失 → 乘以lambda_thin权重。对于thick_gt损失：生成图像（fake_B） → 将thick_gt从[-1,1]归一化到[0,1]作为掩膜 → 应用掩膜提取完整血管区域 → 与标注图像的血管区域计算L1损失 → 乘以lambda_thick权重。

### 训练优化策略

**优化器配置：**
1. **类型**: Adam优化器
2. **学习率**: 0.0002（初始值）
3. **Beta参数**: (0.5, 0.999)
4. **学习率调度**: 线性衰减
   1. 前200个epoch保持0.0002
   2. 后200个epoch线性衰减到0

**训练技巧：**
1. **批量归一化**: 加速收敛，提高训练稳定性
2. **标签平滑**: 判别器训练时使用标签平滑，提高泛化能力
3. **历史图像池**: 存储历史生成的图像，用于判别器训练（本项目未使用）

## 核心代码

### 数据集加载器

```python
# data/rose_dataset.py

class ROSEDataset(BaseDataset):
    """ROSE数据集类，用于加载ROSE-1和ROSE-2的OCTA图像数据。"""
    
    def __init__(self, opt):
        """初始化数据集类。"""
        BaseDataset.__init__(self, opt)
        
        # 获取数据集版本和模式配置
        self.rose_version = getattr(opt, 'rose_version', '1')
        self.rose_mode = opt.rose_mode
        self.use_thin_gt = getattr(opt, 'use_thin_gt', False)
        self.use_thick_gt = getattr(opt, 'use_thick_gt', False)
        
        # 根据数据集版本构建路径
        if self.rose_version == "2":
            phase_dir = os.path.join(opt.dataroot, "ROSE-2", opt.phase)
            self.dir_img = os.path.join(phase_dir, "original")
            self.dir_gt = os.path.join(phase_dir, "gt")
        else:
            mode_dir = os.path.join(opt.dataroot, "ROSE-1", self.rose_mode, opt.phase)
            self.dir_img = os.path.join(mode_dir, "img")
            self.dir_gt = os.path.join(mode_dir, "gt")
            self.dir_thin_gt = os.path.join(mode_dir, "thin_gt")
            self.dir_thick_gt = os.path.join(mode_dir, "thick_gt")
    
    def __getitem__(self, index):
        """获取指定索引的数据样本。"""
        base_name = self.img_paths[index]
        
        # 加载输入图像和标注图像
        img_path = os.path.join(self.dir_img, f"{base_name}{self.img_ext}")
        gt_path = os.path.join(self.dir_gt, f"{base_name}{self.gt_ext}")
        
        A = self.load_image(img_path)
        B = self.load_image(gt_path)
        
        # 加载可选的标注掩膜（仅ROSE-1支持）
        thin_gt = None
        thick_gt = None
        if self.rose_version == "1":
            if self.use_thin_gt and self.has_thin_gt:
                thin_gt_path = os.path.join(self.dir_thin_gt, f"{base_name}{self.gt_ext}")
                thin_gt = self.load_image(thin_gt_path)
            
            if self.use_thick_gt and self.has_thick_gt:
                thick_gt_path = os.path.join(self.dir_thick_gt, f"{base_name}{self.gt_ext}")
                thick_gt = self.load_image(thick_gt_path)
        
        # 应用图像变换
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)
        
        A = A_transform(A)
        B = B_transform(B)
        
        # 构建返回字典
        result = {"A": A, "B": B, "A_paths": img_path, "B_paths": gt_path}
        if thin_gt is not None:
            thin_transform = get_transform(self.opt, transform_params, grayscale=True)
            result["thin_gt"] = thin_transform(thin_gt)
        if thick_gt is not None:
            thick_transform = get_transform(self.opt, transform_params, grayscale=True)
            result["thick_gt"] = thick_transform(thick_gt)
        
        return result
```

### 模型实现

```python
# models/pix2pix_model.py

class Pix2PixModel(BaseModel):
    """图像增强模型类，用于学习从输入图像到输出图像的映射。"""
    
    def backward_G(self):
        """计算生成器的GAN损失、L1损失和结构约束损失"""
        # GAN损失：生成图像应该欺骗判别器
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # L1损失：生成图像应该接近真实标注
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        
        # 结构约束损失（如果可用）
        loss_G_thin = torch.tensor(0.0).to(self.device)
        loss_G_thick = torch.tensor(0.0).to(self.device)
        
        # thin_gt损失：约束骨架化血管结构
        if self.thin_gt is not None and hasattr(self.opt, 'use_thin_gt') and self.opt.use_thin_gt:
            thin_mask = (self.thin_gt + 1.0) / 2.0
            fake_masked = self.fake_B * thin_mask
            real_masked = self.real_B * thin_mask
            loss_G_thin = self.criterionL1(fake_masked, real_masked) * self.opt.lambda_thin
            self.loss_G_thin = loss_G_thin
        
        # thick_gt损失：约束完整血管区域
        if self.thick_gt is not None and hasattr(self.opt, 'use_thick_gt') and self.opt.use_thick_gt:
            thick_mask = (self.thick_gt + 1.0) / 2.0
            fake_masked = self.fake_B * thick_mask
            real_masked = self.real_B * thick_mask
            loss_G_thick = self.criterionL1(fake_masked, real_masked) * self.opt.lambda_thick
            self.loss_G_thick = loss_G_thick
        
        # 合并所有损失并计算梯度
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + loss_G_thin + loss_G_thick
        self.loss_G.backward()
```

### 设备自动选择

```python
# util/util.py

def init_ddp():
    """初始化训练/测试设备。
    
    自动选择CUDA、MPS（Apple Silicon）或CPU设备。
    
    优先级顺序:
    1. CUDA（如果可用且在DDP模式下）
    2. CUDA（如果可用）
    3. MPS（如果Apple Silicon可用）
    4. CPU（后备选项）
    """
    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        print(f"Initialized with device {device} (DDP mode)")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        print(f"Initialized with device {device} (CUDA: {torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Initialized with device {device} (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Initialized with device {device} (CPU mode - training will be slow)")
    
    return device
```

## 使用说明

### 环境要求

1. Python >= 3.9
2. PyTorch >= 2.0
3. 其他依赖：numpy, Pillow, opencv-python, scikit-image

### 安装依赖

手动安装：
```bash
# Mac/Linux
pip3 install torch torchvision numpy Pillow opencv-python scikit-image matplotlib dominate

# Windows
pip install torch torchvision numpy Pillow opencv-python scikit-image matplotlib dominate
```

### 数据集准备

确保数据集目录结构如下：

```
Dataset/
    ROSE/
        ROSE-1/
            SVC/
                train/
                    img/
                    gt/
                    thin_gt/  (可选)
                    thick_gt/ (可选)
                test/
                    ...
            DVC/
                ...
            SVC_DVC/
                ...
        ROSE-2/
            train/
                original/
                gt/
            test/
                ...
```

### 训练模型

#### 方法1：交互式训练脚本（推荐）

![image-20251225140431403](/Users/duchenyi/Library/Application Support/typora-user-images/image-20251225140431403.png)

**Mac/Linux系统：**
```bash
cd pytorch-CycleGAN-and-pix2pix
python3 train_interactive.py
```

**Windows系统：**
```cmd
cd pytorch-CycleGAN-and-pix2pix
python train_interactive.py
```

脚本会引导：
1. 选择数据集（ROSE-1 / ROSE-2 / 合并）
2. 选择ROSE-1的模式（如果选择ROSE-1）
3. 配置训练参数
4. 确认并开始训练

#### 方法2：命令行训练

**训练ROSE-1 SVC模式：**

Mac/Linux系统：
```bash
cd pytorch-CycleGAN-and-pix2pix
python3 train.py \
    --dataroot ../Dataset/ROSE \
    --name rose_svc_pix2pix \
    --model pix2pix \
    --dataset_mode rose \
    --rose_version 1 \
    --rose_mode SVC \
    --use_thin_gt \
    --use_thick_gt \
    --netG unet_256 \
    --direction AtoB \
    --lambda_L1 100.0 \
    --lambda_thin 10.0 \
    --lambda_thick 10.0 \
    --batch_size 4 \
    --load_size 286 \
    --crop_size 256 \
    --n_epochs 200 \
    --n_epochs_decay 200 \
    --lr 0.0002 \
    --norm batch \
    --input_nc 1 \
    --output_nc 1
```

Windows系统（使用 `^` 作为换行符）：
```cmd
cd pytorch-CycleGAN-and-pix2pix
python train.py ^
    --dataroot ../Dataset/ROSE ^
    --name rose_svc_pix2pix ^
    --model pix2pix ^
    --dataset_mode rose ^
    --rose_version 1 ^
    --rose_mode SVC ^
    --use_thin_gt ^
    --use_thick_gt ^
    --netG unet_256 ^
    --direction AtoB ^
    --lambda_L1 100.0 ^
    --lambda_thin 10.0 ^
    --lambda_thick 10.0 ^
    --batch_size 4 ^
    --load_size 286 ^
    --crop_size 256 ^
    --n_epochs 200 ^
    --n_epochs_decay 200 ^
    --lr 0.0002 ^
    --norm batch ^
    --input_nc 1 ^
    --output_nc 1
```

**训练ROSE-2 SVP模式：**

Mac/Linux系统：
```bash
python3 train.py \
    --dataroot ../Dataset/ROSE \
    --name rose2_svp_pix2pix \
    --model pix2pix \
    --dataset_mode rose \
    --rose_version 2 \
    --rose_mode SVP \
    --netG unet_256 \
    --direction AtoB \
    --lambda_L1 100.0 \
    --batch_size 4 \
    --n_epochs 200 \
    --input_nc 1 \
    --output_nc 1
```

Windows系统：
```cmd
python train.py --dataroot ../Dataset/ROSE --name rose2_svp_pix2pix --model pix2pix --dataset_mode rose --rose_version 2 --rose_mode SVP --netG unet_256 --direction AtoB --lambda_L1 100.0 --batch_size 4 --n_epochs 200 --input_nc 1 --output_nc 1
```

#### 方法3：使用训练脚本（仅Mac/Linux）

```bash
cd pytorch-CycleGAN-and-pix2pix
bash scripts/train_rose.sh
```

### 测试模型

**Mac/Linux系统：**
```bash
cd pytorch-CycleGAN-and-pix2pix
python3 test.py \
    --dataroot ../Dataset/ROSE \
    --name rose_svc_pix2pix \
    --model pix2pix \
    --dataset_mode rose \
    --rose_version 1 \
    --rose_mode SVC \
    --direction AtoB \
    --netG unet_256 \
    --input_nc 1 \
    --output_nc 1
```

**Windows系统：**
```cmd
cd pytorch-CycleGAN-and-pix2pix
python test.py --dataroot ../Dataset/ROSE --name rose_svc_pix2pix --model pix2pix --dataset_mode rose --rose_version 1 --rose_mode SVC --direction AtoB --netG unet_256 --input_nc 1 --output_nc 1
```

### 推理自定义图像

#### 方法1：交互式推理脚本（推荐）

![image-20251225140451139](/Users/duchenyi/Library/Application Support/typora-user-images/image-20251225140451139.png)

交互式脚本可以灵活选择模型、输入文件和输出目录，适合处理不同来源的图像。

**Mac系统使用方法：**

1. 打开终端（Terminal），进入项目目录：
```bash
cd /Users/duchenyi/PycharmProjects/ROSE
```

2. 运行交互式脚本：
```bash
python3 inference_interactive.py
```

3. 按提示操作：
   1. **选择模型**：脚本会列出所有可用的训练模型，输入数字选择（例如输入 `1` 选择 `rose_svc_pix2pix`）
   2. **选择输入源**：
      1. 输入 `1`：使用默认test文件夹
      2. 输入 `2`：指定自定义文件夹（可以直接输入路径，也可以拖拽文件夹到终端）
      3. 输入 `3`：指定单个图像文件（可以直接输入路径，也可以拖拽文件到终端）
   3. **选择输出目录**：
      1. 输入 `1`：使用默认输出目录（`test_results/`）
      2. 输入 `2`：指定自定义输出目录（可以直接输入路径，也可以拖拽文件夹到终端）
   4. **确认处理**：输入 `y` 开始处理

4. 处理完成后，结果保存在输出目录中：
   1. `*_enhanced.png`：增强后的图像
   2. `*_comparison.png`：原始图像与增强图像的对比图
   3. `overview.png`：所有图像的概览图（如果处理了多张图像）

**Windows系统使用方法：**

1. 打开命令提示符（CMD）或PowerShell，进入项目目录：
```cmd
cd C:\Users\YourName\PycharmProjects\ROSE
```

2. 运行交互式脚本：
```cmd
python inference_interactive.py
```

3. 按提示操作：
   1. **选择模型**：输入数字选择模型（例如输入 `1`）
   2. **选择输入源**：
      1. 输入 `1`：使用默认test文件夹
      2. 输入 `2`：指定自定义文件夹（输入完整路径，例如 `C:\Users\YourName\Desktop\images`）
      3. 输入 `3`：指定单个图像文件（输入完整路径，例如 `C:\Users\YourName\Desktop\image.tif`）
   3. **选择输出目录**：
      1. 输入 `1`：使用默认输出目录
      2. 输入 `2`：指定自定义输出目录（输入完整路径）
   4. **确认处理**：输入 `y` 开始处理

**路径输入提示：**
1. Mac系统：支持拖拽文件夹/文件到终端，路径会自动填入
2. Windows系统：需要手动输入完整路径，或使用相对路径（相对于当前工作目录）
3. 支持相对路径：例如 `./test` 或 `../images`
4. 支持绝对路径：例如 `/Users/name/images`（Mac）或 `C:\Users\name\images`（Windows）

**使用示例：**

假设要处理你自己提供的图像文件夹：

1. Mac用户：
   1. 运行脚本后，选择输入源时输入 `2`
   2. 将图像文件夹拖拽到终端，路径自动填入
   3. 或手动输入路径：`/Users/name/客户图像`

2. Windows用户：
   1. 运行脚本后，选择输入源时输入 `2`
   2. 手动输入完整路径：`C:\Users\name\客户图像`
   3. 或使用相对路径：`.\客户图像`（如果文件夹在项目目录下）

#### 方法2：直接使用推理脚本

如果只需要处理test文件夹中的图像，可以使用固定配置的脚本：

```bash
# Mac/Linux
python3 inference_custom.py

# Windows
python inference_custom.py
```

脚本会自动处理 `test/` 目录下的所有图像，使用 `rose_svc_pix2pix` 模型，结果保存在 `test_results/` 目录。

**注意**：此脚本使用固定的模型配置，如需使用其他模型或处理其他文件夹，请使用交互式脚本。

### 继续训练

如果需要继续之前的训练，使用 `--continue_train` 和 `--epoch_count` 参数：

**Mac/Linux系统：**
```bash
python3 train.py \
    --dataroot ../Dataset/ROSE \
    --name rose_svc_pix2pix \
    --continue_train \
    --epoch_count 41 \
    ... (其他参数与之前相同)
```

**Windows系统：**
```cmd
python train.py --dataroot ../Dataset/ROSE --name rose_svc_pix2pix --continue_train --epoch_count 41 ... (其他参数与之前相同)
```

## 主要参数说明

### 数据集参数

1. `--rose_version`: 数据集版本，"1" 或 "2"
2. `--rose_mode`: 血管层模式，"SVC"、"DVC"、"SVC_DVC"（ROSE-1）或 "SVP"（ROSE-2）
3. `--use_thin_gt`: 使用骨架化血管标注进行结构约束（仅ROSE-1）
4. `--use_thick_gt`: 使用完整血管掩膜进行结构约束（仅ROSE-1）

### 训练参数

1. `--lambda_L1`: L1损失权重（默认：100.0）
2. `--lambda_thin`: thin_gt结构损失权重（默认：10.0）
3. `--lambda_thick`: thick_gt结构损失权重（默认：10.0）
4. `--batch_size`: 批次大小（默认：1，建议：4）
5. `--n_epochs`: 训练轮数（默认：100）
6. `--lr`: 学习率（默认：0.0002）

### 网络参数

1. `--netG`: 生成器架构（默认：unet_256）
2. `--input_nc`: 输入通道数（灰度图像为1）
3. `--output_nc`: 输出通道数（灰度图像为1）

## 输出文件

训练过程中会生成：

1. **模型检查点**: `checkpoints/{实验名称}/`
   1. `{epoch}_net_G.pth`: 生成器权重
   2. `{epoch}_net_D.pth`: 判别器权重
   3. `latest_net_G.pth`: 最新生成器权重

2. **可视化页面**: `checkpoints/{实验名称}/web/index.html`
   1. 实时显示训练过程中的输入、输出和标注图像

3. **测试结果**: `results/{实验名称}/latest_test/`
   1. 测试图像的增强结果

## 项目结构

```
ROSE/
├── pytorch-CycleGAN-and-pix2pix/    # 主项目目录
│   ├── data/
│   │   └── rose_dataset.py          # ROSE数据集加载器
│   ├── models/
│   │   └── pix2pix_model.py         # 图像增强模型实现
│   ├── util/
│   │   └── util.py                  # 工具函数（设备选择等）
│   ├── train.py                     # 训练脚本
│   ├── test.py                      # 测试脚本
│   ├── train_interactive.py         # 交互式训练脚本
│   └── scripts/
│       ├── train_rose.sh            # ROSE训练脚本
│       └── test_rose.sh            # ROSE测试脚本
├── Dataset/
│   └── ROSE/                        # ROSE数据集
│       ├── ROSE-1/                  # ROSE-1数据集
│       └── ROSE-2/                  # ROSE-2数据集
├── inference_custom.py              # 自定义图像推理脚本（固定配置）
├── inference_interactive.py         # 交互式推理脚本（推荐）
└── README.md                        # 本文档
```


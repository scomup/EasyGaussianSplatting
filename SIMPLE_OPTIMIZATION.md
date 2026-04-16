# 简化3DGS用于相机位姿优化

## 核心思想

通过极大限制3DGS的自由度，强制模型通过优化相机位姿来拟合图像，而不是通过调整高斯球的形状和位置来"伪造"不同视角。

## 简化模型

### 固定参数（完全不优化）
1. **位置 (pws)** - 3D坐标固定
2. **旋转 (rots)** - 固定为单位四元数 [1,0,0,0]
3. **高阶球谐 (high-order SH)** - 固定为0

### 可优化参数
1. **相机位姿 (qcw, tcw)** - 主要优化目标
2. **RGB颜色 (colors)** - 直接RGB，不用球谐
3. **透明度 (alphas)** - 控制可见性
4. **各向同性大小 (scales)** - 单个标量，所有方向相同

## CUDA层面优化

### 跳过的梯度计算

根据链式法则分析，以下梯度完全不需要计算：

1. **dcov3d_drots** - 旋转固定，不需要
2. **dcolor_dpws** - 位置固定，不需要  
3. **dcolor_dshs (高阶)** - 只用DC分量

### 简化的kernel

#### `computeCov3D_simple`
- 输入：各向同性scales (N,) 而不是 (N,3)
- 输出：简化的3D协方差矩阵（对角矩阵 s²I）
- 梯度：只计算 dcov3d_dscales (6维，对角+0)

```cuda
// 原始：Sigma = R * diag(s) * R^T  (需要旋转矩阵)
// 简化：Sigma = s² * I  (单位矩阵)
```

#### `computeCov2D_simple`
- 使用简化的3D协方差
- 仍需计算 dcov2d_dpcs（用于相机梯度）
- dcov2d_dscales 简化为3维（2D协方差的3个独立元素）

#### `sh2Color_simple` 
- 直接使用RGB颜色，不需要球谐计算
- 完全跳过：
  - 视角依赖计算 (twc - pws)
  - 高阶SH系数
  - dcolor_dpws（位置固定）

### 梯度流程

```
Loss
  ↓
dloss_dgamma
  ↓
├─→ dloss_dcolors → dloss_drgb (直接RGB)
├─→ dloss_dalphas (直接)
├─→ dloss_dscales (简化)
└─→ dloss_dus → dloss_dpcs → dloss_dqcw, dloss_dtcw (相机)
```

## 使用方法

### 1. 编译简化CUDA kernel

```bash
cd gsplatcu
python setup.py install
```

### 2. 运行简化训练

```bash
python train_simple_pose.py --path /path/to/dataset
```

### 3. 查看结果

输出文件兼容标准3DGS viewer：
- `data/simple_final.npy` - 转换为完整格式的3DGS参数
- `data/camera_params_simple.npy` - 优化后的相机参数

## 性能优势

1. **计算速度** - 跳过大量不必要的梯度计算
2. **内存占用** - 不存储position和rotation的梯度
3. **优化质量** - 限制自由度，强制优化相机位姿

## 数学细节

### 各向同性高斯

原始3DGS：
```
Sigma = R * diag(sx, sy, sz) * R^T
```

简化版本：
```
Sigma = s² * I
其中 s 是标量
```

### 梯度简化

原始反向传播链：
```
dloss/drot = dloss/dcov3d * dcov3d/drot  ❌ 不需要
dloss/dpws = dloss/dcolor * dcolor/dpws  ❌ 不需要
```

简化后：
```
dloss/ds = dloss/dcov3d * dcov3d/ds  ✓ 需要（简化版）
dloss/drgb = dloss/dcolor * I        ✓ 直接传递
dloss/dqcw = dloss/dpcs * dpcs/dqcw  ✓ 相机位姿
```

## 限制

- 不适合标准3DGS训练（自由度太低）
- 仅用于相机位姿优化
- 输出虽然兼容viewer，但质量会比完整3DGS差
- 适用场景：相机标定不准确，需要fine-tune位姿

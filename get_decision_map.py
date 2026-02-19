#!/usr/bin/env python
# encoding: utf-8

import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from neural_renderer import ARNet, IUNet
from classical_renderer.scatter import ModuleRenderScatter 

# ================= 全局参数配置 =================
# 模型权重路径
ARNET_CHECKPOINT_PATH = './checkpoints/arnet.pth'
IUNET_CHECKPOINT_PATH = './checkpoints/iunet.pth'

# 渲染控制参数
DEFOCUS_SCALE = 5.0  
GAMMA_MIN = 1.0       
GAMMA_MAX = 5.0       
K = 30.0              
GAMMA = 4.0

DISP_FOCUS_FG=0.8
DISP_FOCUS_BG=0.2
# ================= Trimap 决策图控制参数 =================
# 当 defocus 的绝对值 <= 该值时，认为是“肯定对上焦”，渲染为白色 (255)
TRIMAP_IN_FOCUS_TH = 0.6  
# 当 defocus 的绝对值 >= 该值时，认为是“肯定没对上焦”，渲染为黑色 (0)
# 介于 IN_FOCUS_TH 和 OUT_FOCUS_TH 之间的，渲染为灰色 (128)
TRIMAP_OUT_FOCUS_TH = 0.9 
# =========================================================

def gaussian_blur(x, r, sigma=None):
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r) + 1), torch.arange(-int(r), int(r) + 1), indexing='ij')
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2*r+1, 2*r+1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode='replicate')
    x = F.conv2d(x, weight=kernel, padding=0)
    return x

def pipeline(classical_renderer, arnet, iunet, image, defocus, gamma):
    bokeh_classical, defocus_dilate = classical_renderer(image**gamma, defocus*DEFOCUS_SCALE)
    bokeh_classical = bokeh_classical ** (1/gamma)
    defocus_dilate = defocus_dilate / DEFOCUS_SCALE
    gamma_norm = (gamma - GAMMA_MIN) / (GAMMA_MAX - GAMMA_MIN)
    adapt_scale = max(defocus.abs().max().item(), 1)

    image_re = F.interpolate(image, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
    defocus_re = 1 / adapt_scale * F.interpolate(defocus, scale_factor=1/adapt_scale, mode='bilinear', align_corners=True)
    bokeh_neural, error_map = arnet(image_re, defocus_re, gamma_norm)
    error_map = F.interpolate(error_map, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
    bokeh_neural.clamp_(0, 1e5)

    scale = -1
    for scale in range(int(np.log2(adapt_scale))):
        ratio = 2**(scale+1) / adapt_scale
        h_re, w_re = int(ratio * image.shape[2]), int(ratio * image.shape[3])
        image_re = F.interpolate(image, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_re = ratio * F.interpolate(defocus, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_dilate_re = ratio * F.interpolate(defocus_dilate, size=(h_re, w_re), mode='bilinear', align_corners=True)
        bokeh_neural_refine = iunet(image_re, defocus_re.clamp(-1, 1), bokeh_neural, gamma_norm).clamp(0, 1e5)
        mask = gaussian_blur(((defocus_dilate_re < 1) * (defocus_dilate_re > -1)).float(), 0.005 * (defocus_dilate_re.shape[2] + defocus_dilate_re.shape[3]))
        bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(h_re, w_re), mode='bilinear', align_corners=True)

    bokeh_neural_refine = iunet(image, defocus.clamp(-1, 1), bokeh_neural, gamma_norm).clamp(0, 1e5)
    mask = gaussian_blur(((defocus_dilate < 1) * (defocus_dilate > -1)).float(), 0.005 * (defocus_dilate.shape[2] + defocus_dilate.shape[3]))
    bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)

    bokeh_pred = bokeh_classical * (1 - error_map) + bokeh_neural * error_map
    return bokeh_pred.clamp(0, 1)

# =========== 新增：生成 Trimap 的核心函数 ===========
def generate_trimap(defocus_map, in_th, out_th):
    abs_defocus = np.abs(defocus_map)
    trimap = np.zeros_like(abs_defocus, dtype=np.uint8) # 默认全黑 (0，绝对没对上焦)
    
    # 过渡/不确定区域 (128，灰色)
    uncertain_mask = (abs_defocus > in_th) & (abs_defocus < out_th)
    trimap[uncertain_mask] = 128
    
    # 肯定对上焦的区域 (255，白色)
    in_focus_mask = abs_defocus <= in_th
    trimap[in_focus_mask] = 255
    
    return trimap
# ====================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Trimap Focus Decision Maps')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input RGB image')
    parser.add_argument('--disp_path', type=str, required=True, help='Path to input disparity/mask image')
    parser.add_argument('--output', type=str, default='./outputs_decision', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classical_renderer = ModuleRenderScatter().to(device)
    arnet = ARNet(2, 5, 4, 128, 3, False, 'distinct_source', False, 'elu').to(device)
    iunet = IUNet(2, 8, 3, 64, 3, False, 'distinct_source', False, 'elu').to(device)

    # === 修改处：增加 weights_only=False ===
    arnet.load_state_dict(torch.load(ARNET_CHECKPOINT_PATH, map_location=device, weights_only=False)['model'])
    iunet.load_state_dict(torch.load(IUNET_CHECKPOINT_PATH, map_location=device, weights_only=False)['model'])
    arnet.eval()
    iunet.eval()

    save_root = os.path.join(args.output, os.path.splitext(os.path.basename(args.image_path))[0])
    os.makedirs(save_root, exist_ok=True)

    image = cv2.imread(args.image_path).astype(np.float32) / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    disp = np.float32(cv2.imread(args.disp_path, cv2.IMREAD_GRAYSCALE))
    disp = (disp - disp.min()) / (disp.max() - disp.min())

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)

    # ================== 场景 1：对焦在前景 ==================
    print("Processing Foreground Focus...")
    disp_focus_fg = DISP_FOCUS_FG
    defocus_fg = K * (disp - disp_focus_fg) / DEFOCUS_SCALE
    
    # 1. 生成 Trimap 决策图
    trimap_fg = generate_trimap(defocus_fg, TRIMAP_IN_FOCUS_TH, TRIMAP_OUT_FOCUS_TH)

    defocus_fg_tensor = torch.from_numpy(defocus_fg).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        bokeh_pred_fg = pipeline(classical_renderer, arnet, iunet, image_tensor, defocus_fg_tensor, GAMMA)
    bokeh_pred_fg_np = bokeh_pred_fg[0].cpu().permute(1, 2, 0).numpy()


    # ================== 场景 2：对焦在背景 ==================
    print("Processing Background Focus...")
    disp_focus_bg = DISP_FOCUS_BG
    defocus_bg = K * (disp - disp_focus_bg) / DEFOCUS_SCALE
    
    # 2. 生成 Trimap 决策图
    trimap_bg = generate_trimap(defocus_bg, TRIMAP_IN_FOCUS_TH, TRIMAP_OUT_FOCUS_TH)

    defocus_bg_tensor = torch.from_numpy(defocus_bg).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        bokeh_pred_bg = pipeline(classical_renderer, arnet, iunet, image_tensor, defocus_bg_tensor, GAMMA)
    bokeh_pred_bg_np = bokeh_pred_bg[0].cpu().permute(1, 2, 0).numpy()

    # ================== 结果保存 ==================
    # 保存原始 Trimap
    cv2.imwrite(os.path.join(save_root, 'trimap_focus_fg.png'), trimap_fg)
    cv2.imwrite(os.path.join(save_root, 'trimap_focus_bg.png'), trimap_bg)

    # 融合两个 Trimap（原始）
    trimap_fused = np.maximum(trimap_fg, trimap_bg)
    cv2.imwrite(os.path.join(save_root, 'trimap_fused_raw.png'), trimap_fused)

    # 形态学处理：先开运算后闭运算，多次迭代平滑 trimap
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 先开运算（腐蚀+膨胀）去除小的白色噪点
    trimap_temp = cv2.morphologyEx(trimap_fused, cv2.MORPH_OPEN, kernel)
    # 再闭运算（膨胀+腐蚀）填充黑色空洞
    trimap_fused_smooth = cv2.morphologyEx(trimap_temp, cv2.MORPH_CLOSE, kernel)
    
    # 可选：重复一次以获得更好的平滑效果
    trimap_temp = cv2.morphologyEx(trimap_fused_smooth, cv2.MORPH_OPEN, kernel)
    trimap_fused_smooth = cv2.morphologyEx(trimap_temp, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite(os.path.join(save_root, 'trimap_fused_smooth.png'), trimap_fused_smooth)

    # 保存渲染好的虚化图
    cv2.imwrite(os.path.join(save_root, 'bokeh_focus_fg.jpg'), bokeh_pred_fg_np[..., ::-1] * 255)
    cv2.imwrite(os.path.join(save_root, 'bokeh_focus_bg.jpg'), bokeh_pred_bg_np[..., ::-1] * 255)

    # 计算与原图的差距（绝对差值）
    diff_fg = np.abs(image - bokeh_pred_fg_np)
    diff_bg = np.abs(image - bokeh_pred_bg_np)
    
    # 保存差距图（增强显示以便观察）
    cv2.imwrite(os.path.join(save_root, 'diff_focus_fg.jpg'), (diff_fg[..., ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_root, 'diff_focus_bg.jpg'), (diff_bg[..., ::-1] * 255).astype(np.uint8))

    print(f"Done! Check results in: {save_root}")

if __name__ == '__main__':
    main()
#!/usr/bin/env python
# encoding: utf-8

import os
import io
import glob
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import webdataset as wds

from neural_renderer import ARNet, IUNet
from classical_renderer.scatter import ModuleRenderScatter

# ================= 全局参数配置 =================
ARNET_CHECKPOINT_PATH = './checkpoints/arnet.pth'
IUNET_CHECKPOINT_PATH = './checkpoints/iunet.pth'

DEFOCUS_SCALE = 5.0
GAMMA_MIN = 1.0
GAMMA_MAX = 5.0
K = 30.0
GAMMA = 4.0

DISP_FOCUS_FG = 0.8
DISP_FOCUS_BG = 0.2

TRIMAP_IN_FOCUS_TH = 0.6
TRIMAP_OUT_FOCUS_TH = 0.9
# =================================================


def gaussian_blur(x, r, sigma=None):
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(
        torch.arange(-int(r), int(r) + 1),
        torch.arange(-int(r), int(r) + 1),
        indexing='ij'
    )
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2 * r + 1, 2 * r + 1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode='replicate')
    x = F.conv2d(x, weight=kernel, padding=0)
    return x


def pipeline(classical_renderer, arnet, iunet, image, defocus, gamma):
    bokeh_classical, defocus_dilate = classical_renderer(image ** gamma, defocus * DEFOCUS_SCALE)
    bokeh_classical = bokeh_classical ** (1 / gamma)
    defocus_dilate = defocus_dilate / DEFOCUS_SCALE
    gamma_norm = (gamma - GAMMA_MIN) / (GAMMA_MAX - GAMMA_MIN)
    adapt_scale = max(defocus.abs().max().item(), 1)

    image_re = F.interpolate(image, scale_factor=1 / adapt_scale, mode='bilinear', align_corners=True)
    defocus_re = 1 / adapt_scale * F.interpolate(defocus, scale_factor=1 / adapt_scale, mode='bilinear', align_corners=True)
    bokeh_neural, error_map = arnet(image_re, defocus_re, gamma_norm)
    error_map = F.interpolate(error_map, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)
    bokeh_neural.clamp_(0, 1e5)

    for scale in range(int(np.log2(adapt_scale))):
        ratio = 2 ** (scale + 1) / adapt_scale
        h_re, w_re = int(ratio * image.shape[2]), int(ratio * image.shape[3])
        image_re = F.interpolate(image, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_re = ratio * F.interpolate(defocus, size=(h_re, w_re), mode='bilinear', align_corners=True)
        defocus_dilate_re = ratio * F.interpolate(defocus_dilate, size=(h_re, w_re), mode='bilinear', align_corners=True)
        bokeh_neural_refine = iunet(image_re, defocus_re.clamp(-1, 1), bokeh_neural, gamma_norm).clamp(0, 1e5)
        mask = gaussian_blur(
            ((defocus_dilate_re < 1) * (defocus_dilate_re > -1)).float(),
            0.005 * (defocus_dilate_re.shape[2] + defocus_dilate_re.shape[3])
        )
        bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(h_re, w_re), mode='bilinear', align_corners=True)

    bokeh_neural_refine = iunet(image, defocus.clamp(-1, 1), bokeh_neural, gamma_norm).clamp(0, 1e5)
    mask = gaussian_blur(
        ((defocus_dilate < 1) * (defocus_dilate > -1)).float(),
        0.005 * (defocus_dilate.shape[2] + defocus_dilate.shape[3])
    )
    bokeh_neural = mask * bokeh_neural_refine + (1 - mask) * F.interpolate(bokeh_neural, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=True)

    bokeh_pred = bokeh_classical * (1 - error_map) + bokeh_neural * error_map
    return bokeh_pred.clamp(0, 1)


def generate_trimap(defocus_map, in_th, out_th):
    abs_defocus = np.abs(defocus_map)
    trimap = np.zeros_like(abs_defocus, dtype=np.uint8)
    trimap[(abs_defocus > in_th) & (abs_defocus < out_th)] = 128
    trimap[abs_defocus <= in_th] = 255
    return trimap


def smooth_trimap(trimap):
    kernel_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    t = cv2.morphologyEx(trimap, cv2.MORPH_OPEN, kernel)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
    t = cv2.morphologyEx(t, cv2.MORPH_OPEN, kernel)
    t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, kernel)
    return t


def pil_to_rgb_float32(pil_img):
    """PIL Image → numpy float32 RGB [0,1]"""
    arr = np.array(pil_img.convert('RGB'), dtype=np.float32) / 255.0
    return arr


def pil_to_gray_float32(pil_img):
    """PIL Image → numpy float32 grayscale, normalized to [0,1]"""
    arr = np.array(pil_img.convert('L'), dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr


def process_sample(sample, classical_renderer, arnet, iunet, device, output_folder, resume):
    key = sample['__key__']
    parts = key.split('/', 1)
    if len(parts) == 2:
        subfolder, file_stem = parts
    else:
        subfolder, file_stem = '', parts[0]

    save_dir = os.path.join(output_folder, subfolder)

    # Resume: skip if all outputs already exist
    if resume:
        expected = [
            os.path.join(save_dir, f'{file_stem}.jpg'),
            os.path.join(save_dir, f'{file_stem}_depth.jpg'),
            os.path.join(save_dir, f'{file_stem}_bokeh_focus_fg.jpg'),
            os.path.join(save_dir, f'{file_stem}_bokeh_focus_bg.jpg'),
            os.path.join(save_dir, f'{file_stem}_trimap_fused_smooth.jpg'),
        ]
        if all(os.path.exists(p) for p in expected):
            return

    pil_image = sample.get('jpg')
    depth_raw = sample.get('jpg;depth')  # bytes: decoder skips non-standard extensions

    if pil_image is None:
        print(f'[WARN] {key}: missing image, skipping')
        return
    if depth_raw is None:
        print(f'[WARN] {key}: missing depth, skipping')
        return

    # Manually decode depth bytes → PIL Image
    if isinstance(depth_raw, bytes):
        pil_depth = Image.open(io.BytesIO(depth_raw))
    else:
        pil_depth = depth_raw  # already decoded (shouldn't happen, but safe)

    os.makedirs(save_dir, exist_ok=True)

    # Convert inputs
    image_np = pil_to_rgb_float32(pil_image)   # H x W x 3, float32 [0,1]
    disp_np  = pil_to_gray_float32(pil_depth)  # H x W, float32 [0,1]

    image_tensor   = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # Foreground focus
    defocus_fg = K * (disp_np - DISP_FOCUS_FG) / DEFOCUS_SCALE
    trimap_fg  = generate_trimap(defocus_fg, TRIMAP_IN_FOCUS_TH, TRIMAP_OUT_FOCUS_TH)
    defocus_fg_tensor = torch.from_numpy(defocus_fg).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        bokeh_fg = pipeline(classical_renderer, arnet, iunet, image_tensor, defocus_fg_tensor, GAMMA)
    bokeh_fg_np = bokeh_fg[0].cpu().permute(1, 2, 0).numpy()

    # Background focus
    defocus_bg = K * (disp_np - DISP_FOCUS_BG) / DEFOCUS_SCALE
    trimap_bg  = generate_trimap(defocus_bg, TRIMAP_IN_FOCUS_TH, TRIMAP_OUT_FOCUS_TH)
    defocus_bg_tensor = torch.from_numpy(defocus_bg).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        bokeh_bg = pipeline(classical_renderer, arnet, iunet, image_tensor, defocus_bg_tensor, GAMMA)
    bokeh_bg_np = bokeh_bg[0].cpu().permute(1, 2, 0).numpy()

    # Fuse and smooth trimap
    trimap_fused        = np.maximum(trimap_fg, trimap_bg)
    trimap_fused_smooth = smooth_trimap(trimap_fused)

    # Save outputs (all jpg)
    # Original image (RGB→BGR for cv2)
    cv2.imwrite(os.path.join(save_dir, f'{file_stem}.jpg'),
                (image_np[..., ::-1] * 255).astype(np.uint8))

    # Depth map (grayscale saved as jpg)
    depth_uint8 = (disp_np * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{file_stem}_depth.jpg'), depth_uint8)

    # Bokeh results (RGB→BGR for cv2)
    cv2.imwrite(os.path.join(save_dir, f'{file_stem}_bokeh_focus_fg.jpg'),
                (bokeh_fg_np[..., ::-1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(save_dir, f'{file_stem}_bokeh_focus_bg.jpg'),
                (bokeh_bg_np[..., ::-1] * 255).astype(np.uint8))

    # Trimap (grayscale, saved as jpg)
    cv2.imwrite(os.path.join(save_dir, f'{file_stem}_trimap_fused_smooth.jpg'),
                trimap_fused_smooth)


def main():
    parser = argparse.ArgumentParser(description='Batch bokeh rendering from WebDataset shards')
    parser.add_argument('--shard_dir', type=str, default='./webdataset_shards',
                        help='Directory containing sa_images-*.tar shards')
    parser.add_argument('--output',    type=str, default='./outputs_wds',
                        help='Root output directory')
    parser.add_argument('--device',    type=str, default='',
                        help='cuda / cpu (default: auto-detect)')
    parser.add_argument('--resume',    action='store_true',
                        help='Skip samples whose output files already exist')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f'Using device: {device}')

    # Load models once
    classical_renderer = ModuleRenderScatter().to(device)
    arnet = ARNet(2, 5, 4, 128, 3, False, 'distinct_source', False, 'elu').to(device)
    iunet = IUNet(2, 8, 3, 64, 3, False, 'distinct_source', False, 'elu').to(device)

    arnet.load_state_dict(torch.load(ARNET_CHECKPOINT_PATH, map_location=device, weights_only=False)['model'])
    iunet.load_state_dict(torch.load(IUNET_CHECKPOINT_PATH, map_location=device, weights_only=False)['model'])
    arnet.eval()
    iunet.eval()

    shards = sorted(glob.glob(os.path.join(args.shard_dir, 'sa_images-*.tar')))
    if not shards:
        print(f'[ERROR] No shards found in {args.shard_dir}')
        return
    print(f'Found {len(shards)} shard(s)')

    dataset = wds.WebDataset(shards, shardshuffle=False).decode('pil')

    for sample in tqdm(dataset, desc='Processing', unit='img'):
        process_sample(
            sample,
            classical_renderer, arnet, iunet,
            device,
            args.output,
            args.resume,
        )

    print(f'Done. Results saved to: {args.output}')


if __name__ == '__main__':
    main()

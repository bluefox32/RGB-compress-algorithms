import cv2
import numpy as np
import os

def enhance_color(channel, factor=1.1):
    """Enhance the color of a given channel by a specified factor."""
    return np.clip(channel * factor, 0, 255).astype(np.uint8)

def smooth_channel(channel):
    """Smooth the channel using GaussianBlur for simplicity."""
    return cv2.GaussianBlur(channel, (5, 5), 0)

def process_image_sequence(input_dir, output_dir, oversampling_factor):
    # 入力ディレクトリの画像ファイルリストを取得
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_number = 0
    
    for image_file in image_files:
        frame = cv2.imread(os.path.join(input_dir, image_file))
        
        # フレームをRGBチャンネルに分割
        b, g, r = cv2.split(frame)
        
        # RGBチャンネルをオーバーサンプリング（12倍）して平滑化
        b = smooth_channel(b * 12)
        g = smooth_channel(g * 12)
        r = smooth_channel(r * 12)
        
        # 平滑化されたチャンネルにデノイズを適用
        b = cv2.fastNlMeansDenoising(b, None, 10, 7, 21)
        g = cv2.fastNlMeansDenoising(g, None, 10, 7, 21)
        r = cv2.fastNlMeansDenoising(r, None, 10, 7, 21)
        
        # 元のチャンネルと比較して色情報を1.1倍に強化
        b = enhance_color(b)
        g = enhance_color(g)
        r = enhance_color(r)
        
        # チャンネルを結合してフレームを再構成
        enhanced_frame = cv2.merge([b, g, r])
        
        # フレームを出力ディレクトリに保存
        output_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
        cv2.imwrite(output_filename, enhanced_frame)
        
        frame_number += 1
    
    # 処理されたフレーム間の補間を実施
    processed_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    
    for i in range(len(processed_files) - 1):
        prev_frame = cv2.imread(os.path.join(output_dir, processed_files[i]))
        next_frame = cv2.imread(os.path.join(output_dir, processed_files[i + 1]))
        
        # フレーム間の補間フレームを作成
        interpolated_frames = interpolate_frames(prev_frame, next_frame, oversampling_factor)
        
        for j, interp_frame in enumerate(interpolated_frames):
            output_filename = os.path.join(output_dir, f"frame_{frame_number:04d}.png")
            cv2.imwrite(output_filename, interp_frame)
            frame_number += 1

def interpolate_frames(frame1, frame2, num_interpolations):
    """Interpolate between two frames to create additional frames."""
    interpolated_frames = []
    for i in range(1, num_interpolations + 1):
        alpha = i / (num_interpolations + 1)
        interpolated_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        interpolated_frames.append(interpolated_frame)
    return interpolated_frames

# 使用例
process_image_sequence("input_images", "output_images", oversampling_factor=12)
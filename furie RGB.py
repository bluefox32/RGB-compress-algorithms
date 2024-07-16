import pyautogui
import numpy as np
from scipy.fft import fftn
import matplotlib.pyplot as plt
import cv2

def capture_screen():
    # スクリーンキャプチャを取得
    screenshot = pyautogui.screenshot()
    # スクリーンショットをNumPy配列に変換
    frame = np.array(screenshot)
    # BGRからRGBに変換（OpenCVはBGR形式を使用）
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def apply_fft_to_frame(frame):
    # 各カラーチャンネルに対してフーリエ変換を適用
    fft_r = fftn(frame[:, :, 0])
    fft_g = fftn(frame[:, :, 1])
    fft_b = fftn(frame[:, :, 2])
    return fft_r, fft_g, fft_b

def plot_fft_magnitude(fft_data, title):
    magnitude = np.log(np.abs(fft_data) + 1)  # より良い視覚化のために対数スケール
    plt.imshow(np.mean(magnitude, axis=0), cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()

# スクリーンキャプチャを取得
frame = capture_screen()

# フーリエ変換を適用
fft_r, fft_g, fft_b = apply_fft_to_frame(frame)

# 結果をプロット
plot_fft_magnitude(fft_r, "FFT Magnitude - Red Channel")
plot_fft_magnitude(fft_g, "FFT Magnitude - Green Channel")
plot_fft_magnitude(fft_b, "FFT Magnitude - Blue Channel")
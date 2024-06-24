import cv2
import numpy as np
import pandas as pd

def calculate_centroid(mask):
    """Calculate the centroid of the given binary mask."""
    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

def process_video(video_path):
    # 動画を読み込む
    cap = cv2.VideoCapture(video_path)
    
    frame_centroids = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームをRGBチャンネルに分割
        b, g, r = cv2.split(frame)
        
        # 各チャンネルの重心を計算
        r_centroid = calculate_centroid(r)
        g_centroid = calculate_centroid(g)
        b_centroid = calculate_centroid(b)
        
        # フレーム番号を取得
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 重心を記録
        frame_centroids.append({
            'frame': frame_number,
            'r_centroid_x': r_centroid[0], 'r_centroid_y': r_centroid[1],
            'g_centroid_x': g_centroid[0], 'g_centroid_y': g_centroid[1],
            'b_centroid_x': b_centroid[0], 'b_centroid_y': b_centroid[1]
        })
    
    cap.release()
    
    # データをデータフレームに変換
    df = pd.DataFrame(frame_centroids)
    # CSVファイルとして保存
    df.to_csv("rgb_centroids.csv", index=False)
    print("RGB各チャンネルの重心を記録しました。")

# 使用例
process_video("input_video.mp4")
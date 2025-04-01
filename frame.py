import subprocess
import sys

def get_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm未安装，正在安装...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tqdm'])
        from tqdm import tqdm
    return tqdm

tqdm = get_tqdm()
import cv2
import os
from datetime import datetime
from tqdm import tqdm
import subprocess
import sys

# 视频文件路径
video_paths = [
    'video_2024-10-28_10-42-19.mp4', # 添加更多视频文件路径
]

# 输出图片的文件夹
output_folder = 'video_2024-10-28_10-42-19'  # 替换为你想保存图片的文件夹路径

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for video_path in video_paths:
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("无法打开视频")
        exit()

    # 获取视频帧率（FPS）
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取当前日期（年、月、日）
    current_date = datetime.now().strftime('%Y-%m-%d')

    frame_count = 0  # 记录帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数

    # 使用 tqdm 显示进度条
    with tqdm(total=total_frames, desc="处理视频", unit="帧") as pbar:
        # 循环读取视频的每一帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 视频读取完毕

            # 获取系统当前时间（年-月-日 时:分:秒:毫秒）
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]  # 去掉最后的3位微秒，只保留毫秒

            # 保存当前帧为图片，并使用系统时间命名
            frame_filename = os.path.join(output_folder, f'frame_{current_time}_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)

            frame_count += 1
            pbar.update(1)  # 更新进度条

    # 释放视频捕获对象
    cap.release()
    print(f'视频处理完毕，共保存 {frame_count} 帧')


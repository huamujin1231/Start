import os
import re
import shutil
import glob
import json
import cv2
import psutil
import torch
import threading
import sys
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, List, Optional, Dict, Tuple 

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    RESET = '\033[0m'  # 重置颜色
    
def print_colored_menu():
    print(f"{Colors.CYAN}\n菜单选项(-回到菜单)：{Colors.RESET}")
    print(f"{Colors.GREEN}1. 模型操作（训练预测onnx）{Colors.RESET}")
    print(f"{Colors.GREEN}2. 视频识别{Colors.RESET}")
    print(f"{Colors.GREEN}3. 修改索引数{Colors.RESET}")
    print(f"{Colors.GREEN}4. 删除索引数{Colors.RESET}")
    print(f"{Colors.GREEN}5. 批量转换 JSON 文件为 YOLO 格式的 TXT 文件{Colors.RESET}")
    print(f"{Colors.GREEN}6. 查找并移动匹配文件（提取索引文件）{Colors.RESET}")
    print(f"{Colors.GREEN}7. 查找并复制匹配文件（原图标注）{Colors.RESET}")
    print(f"{Colors.GREEN}8. 查找并提取匹配文件（消除同文件）{Colors.RESET}")
    print(f"{Colors.GREEN}9. 查找并删除不匹配文件（消除不同文件名的文件）{Colors.RESET}")    
    #print(f"{Colors.YELLOW}	{Colors.RESET}")
    #print(f"{Colors.BLUE}	{Colors.RESET}")
    #print(f"{Colors.MAGENTA}	{Colors.RESET}")
    print(f"{Colors.RED}0. 退出程序{Colors.RESET}")

# 功能3：修改索引数
def set_working_directory() -> str:
    """设置工作目录为当前目录"""
    current_directory = os.getcwd()
    print(f"工作目录已设置为: {current_directory}")
    return current_directory

def find_index_folders(directory: str) -> List[str]:
    """查找目录下以 '-index' 结尾的文件夹"""
    return [folder for folder in os.listdir(directory) if folder.endswith('-index') and os.path.isdir(os.path.join(directory, folder))]

def select_index_folder(folders: List[str]) -> str:
    """让用户选择需要修改索引数的文件夹"""
    print("找到以下以 '-index' 结尾的文件夹：")
    for idx, folder in enumerate(folders, start=1):
        print(f"{idx}: {folder}")
    
    while True:
        try:
            choice = int(input("请选择要修改的文件夹编号：")) - 1
            if choice == '-':
                os.execv(sys.executable, [sys.executable] + sys.argv)
            if 0 <= choice < len(folders):
                return folders[choice]
            else:
                print("无效的选择，请输入正确的编号。")
        except ValueError:
            print("输入无效，请输入数字。")

def modify_index_in_file(file_path: str, old_index: int, new_index: int):
    """修改文件中的索引数"""
    try:
        # 读取文件内容并修改
        with open(file_path, 'r') as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            modified_line = re.sub(rf"\b{old_index}\b", str(new_index), line)
            modified_lines.append(modified_line)

        # 将修改后的内容写回同名文件
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)
        print(f"文件 {file_path} 中的索引 {old_index} 已修改为 {new_index}")
    
    except Exception as e:
        print(f"修改文件 {file_path} 时发生错误: {e}")

def backup_index_folder(folder_path: str):
    """备份整个 index 文件夹"""
    backup_folder = f"{folder_path}.bak"
    try:
        shutil.copytree(folder_path, backup_folder)
        print(f"文件夹 {folder_path} 已备份为 {backup_folder}")
    except Exception as e:
        print(f"备份文件夹 {folder_path} 时发生错误: {e}")

def get_modify_index() -> Tuple[int, int]:
    """让用户输入要修改的索引数以及修改后的新索引数"""
    try:
        old_index = int(input("请输入需要修改的索引数："))
        new_index = int(input(f"请输入要将索引 {old_index} 修改为的新索引数："))
        return old_index, new_index
    except ValueError:
        print("输入无效，请输入有效的整数。")
        return get_modify_index()

def modify_indices_in_folder(directory: str, old_index: int, new_index: int):
    """遍历目录中的所有 .txt 文件并修改索引数"""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            modify_index_in_file(file_path, old_index, new_index)
# 功能 4：删除索引数
def get_delete_index() -> Tuple[int, int]:
    """让用户输入要删除的索引数行"""
    try:
        target_index = int(input("请输入需要删除的索引数行："))
        return target_index
    except ValueError:
        print(f"{Colors.RED}输入无效，请输入有效的整数。{Colors.RESET}")
        return get_delete_index()

def delete_indices_in_folder(directory: str, target_index: int):
    """遍历目录中的所有 .txt 文件并修改索引数"""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            delete_matching_lines(file_path, target_index)

def delete_matching_lines(file_path, target_index):
    """删除文件中的索引数行"""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()  
        with open(file_path, 'w') as file:
            for line in lines:
                # 以空格分隔行，并获取索引数
                parts = line.split()
                if len(parts) == 0 or parts[0] != str(target_index):
                    # 只写入不匹配的行
                    file.write(line)    
        print(f"文件 {file_path} 中的 {target_index} 索引行已删除")
    except Exception as e:
        print(f"{Colors.RED}删除文件 {file_path} 时发生错误: {e}{Colors.RESET}")    
# 功能 5：批量转换 JSON 文件为 YOLO 格式的 TXT 文件
def convert_labelme_to_yolo(json_path: str, image_path: str, classes: List[str], output_path: str, default_class_index: int) -> None:
    """将 LabelMe 的 JSON 文件转换为 YOLO 格式的 TXT 文件。"""
    try:
        print(f"正在处理文件: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图像文件: {image_path}")
            return
        
        h, w = img.shape[:2]
        print(f"图像尺寸: 宽 {w}, 高 {h}")

        with open(output_path, 'w') as f:
            for shape in data['shapes']:
                label = shape['label']
                if label not in classes:
                    print(f"警告：未找到标签 '{label}'，将其转换为自定义类别。")
                    class_index = default_class_index
                else:
                    class_index = classes.index(label)
                print(f"处理类别: {label}, 类别索引: {class_index}")
		
                points = shape['points']
                if len(points) == 2:
                    (x1, y1), (x2, y2) = points
                else:
                    x_coordinates = [p[0] for p in points]
                    y_coordinates = [p[1] for p in points]
                    x1, y1 = min(x_coordinates), min(y_coordinates)
                    x2, y2 = max(x_coordinates), max(y_coordinates)

                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = abs(x2 - x1) / w
                height = abs(y2 - y1) / h

                f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")
        
        print(f"成功将 {json_path} 转换为 {output_path}")
    
    except Exception as e:
        print(f"转换时发生错误: {e}")

def batch_convert(json_folder: str, image_folder: str, output_folder: str, classes: List[str], default_class_index: int) -> None:
    """批量转换 JSON 文件为 YOLO 格式的 TXT 文件，支持多种图像格式。"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")

    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            image_name_base = json_file.replace('.json', '')

            # 查找对应的图像文件
            image_path = None
            for ext in image_extensions:
                potential_image_path = os.path.join(image_folder, image_name_base + ext)
                if os.path.exists(potential_image_path):
                    image_path = potential_image_path
                    break

            if image_path:
                output_path = os.path.join(output_folder, json_file.replace('.json', '.txt'))
                convert_labelme_to_yolo(json_path, image_path, classes, output_path, default_class_index)
            else:
                print(f"未找到对应的图像文件：{image_name_base}，跳过该文件。")
# 功能6：根据用户输入的索引数查找并移动文件
def find_files_by_class_index(src_folder: str, indices: set) -> List[str]:
    """根据用户输入的索引数查找所有包含这些索引类别的文件"""
    matching_files = []
    
    # 获取源文件夹中的所有 .txt 文件
    txt_files = [f for f in os.listdir(src_folder) if f.endswith('.txt')]
    
    # 遍历所有文本文件
    for file in txt_files:
        file_path = os.path.join(src_folder, file)
        
        try:
            # 打开并读取文件内容
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # 遍历每一行，检查第一位数字是否匹配索引
                for line in lines:
                    line_parts = line.split()
                    if line_parts and line_parts[0].isdigit():
                        class_index = int(line_parts[0])
                        
                        # 如果该类别索引在用户提供的索引集中，标记为匹配
                        if class_index in indices:
                            matching_files.append(file_path)
                            break  # 找到匹配即可退出当前文件的遍历
            
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    return matching_files

def prompt_user_and_move_files(matching_files, dest_folder):
    """询问用户是否要批量复制匹配的文件到目标文件夹"""
    if matching_files:
        # 列出所有找到的匹配文件
        print("找到以下匹配的文件：")
        for file in matching_files:
            print(f"- {os.path.basename(file)}")
        
        # 询问用户是否进行批量复制
        user_input = input(f"是否将这些文件复制到目标文件夹 '{dest_folder}'？(y/n): ").strip().lower()
        
        if user_input == 'y':
            move_files_to_folder(matching_files, dest_folder)
            print(f"已将 {len(matching_files)} 个文件移动到 '{dest_folder}'")
        else:
            print("操作已取消，未复制任何文件。")
    else:
        print("没有找到匹配的文件。")

def move_files_to_folder(files, destination_folder):
    """将文件移动到目标文件夹"""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for file in files:
        file_name = os.path.basename(file)
        dest_path = os.path.join(destination_folder, file_name)
        os.rename(file, dest_path)
        print(f"文件 {file_name} 已移动到 {dest_path}")


# 功能 7：查找并复制匹配文件
def find_and_copy_matching_files(src_folder, dest_folder, target_folder):
    # 获取源文件夹中的文件列表
    src_files = set(os.path.splitext(f)[0] for f in os.listdir(src_folder))  # 只获取文件名部分
    dest_files = set(os.path.splitext(f)[0] for f in os.listdir(dest_folder))  # 只获取文件名部分
    
    # 查找源文件夹和目标文件夹中同名的文件（不考虑扩展名）
    matching_files = src_files.intersection(dest_files)
    
    # 如果找到匹配的文件，将它们从目标文件夹复制到目标文件夹
    if matching_files:
        for file_name in matching_files:
            # 找到源文件夹和目标文件夹中的实际文件路径
            src_file_path = next((os.path.join(src_folder, f) for f in os.listdir(src_folder) if os.path.splitext(f)[0] == file_name), None)
            dest_file_path = next((os.path.join(dest_folder, f) for f in os.listdir(dest_folder) if os.path.splitext(f)[0] == file_name), None)
            target_file_path = os.path.join(target_folder, os.path.basename(dest_file_path))  # 要复制到的目标文件夹路径
            
            # 检查源文件是否存在，并进行复制
            if src_file_path and dest_file_path and os.path.exists(dest_file_path):
                shutil.copy(dest_file_path, target_file_path)
                print(f"文件 {os.path.basename(dest_file_path)} 已成功复制到 {target_folder}")
            else:
                print(f"文件 {file_name} 不存在，无法复制。")
    else:
        print("没有找到匹配的文件。")
# 功能 8：查找并提取匹配文件
def find_and_extract_matching_files(src_folder, dest_folder, target_folder):
    # 获取源文件夹中的文件名（去掉扩展名）
    src_files = {os.path.splitext(f)[0] for f in os.listdir(src_folder)}
    
    # 获取目标文件夹中的文件名（去掉扩展名）
    dest_files = {os.path.splitext(f)[0] for f in os.listdir(dest_folder)}
    
    # 查找源文件夹和目标文件夹中同名的文件
    matching_files = src_files.intersection(dest_files)
    
    # 如果找到匹配的文件，将它们从目标文件夹提取到目标文件夹
    if matching_files:
        for file in matching_files:
            # 查找匹配的所有文件扩展名
            src_matching_files = [f for f in os.listdir(src_folder) if os.path.splitext(f)[0] == file]
            dest_matching_files = [f for f in os.listdir(dest_folder) if os.path.splitext(f)[0] == file]
            
            # 移动目标文件夹中的匹配文件到目标文件夹
            for dest_file in dest_matching_files:
                src_file_path = os.path.join(dest_folder, dest_file)  # 文件在目标文件夹中的路径
                target_file_path = os.path.join(target_folder, dest_file)  # 要移动到的目标文件夹路径
                
                if os.path.exists(src_file_path):
                    shutil.move(src_file_path, target_file_path)
                    print(f"文件 {dest_file} 已成功提取并移动到 {target_folder}")
    else:
        print("没有找到匹配的文件。")
# 功能 9：查找并删除不匹配文件（消除不同文件名的文件）
def find_and_remove_non_matching_files(src_folder, dest_folder):
    # 创建备份文件夹路径
    backup_folder = f"{dest_folder}.bak"
    
    # 确保备份文件夹存在
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    
    # 获取源文件夹中的文件名（去掉扩展名）
    src_files = {os.path.splitext(f)[0] for f in os.listdir(src_folder)}
    
    # 获取目标文件夹中的文件名（去掉扩展名）
    dest_files = {os.path.splitext(f)[0] for f in os.listdir(dest_folder)}
    
    # 查找源文件夹和目标文件夹中相同的文件名
    matching_files = src_files.intersection(dest_files)
    
    # 查找目标文件夹中不匹配的文件名
    non_matching_dest_files = dest_files - matching_files

    # 备份目标文件夹中的所有文件到备份文件夹
    for file in os.listdir(dest_folder):
        src_file_path = os.path.join(dest_folder, file)
        backup_file_path = os.path.join(backup_folder, file)
        shutil.copy(src_file_path, backup_file_path)
        print(f"已备份目标文件夹中的文件: {file} 到 {backup_folder}")

    # 删除目标文件夹中的不匹配文件
    for file in os.listdir(dest_folder):
        file_base = os.path.splitext(file)[0]
        if file_base in non_matching_dest_files:
            dest_file_path = os.path.join(dest_folder, file)
            if os.path.isfile(dest_file_path):
                os.remove(dest_file_path)
                print(f"目标文件夹中不匹配的文件已删除: {file}")
# 新增功能：遍历当前目录下的所有文件夹
def list_folders_in_current_directory() -> list:
    """遍历当前目录下的所有文件夹，并返回文件夹列表。"""
    current_directory = os.getcwd()
    return [folder for folder in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, folder))]

# 新增功能：提供用户选择文件夹，或输入0自定义路径
def select_folder(prompt: str, filter_index_folders: bool = False) -> str:
    """提供用户选择文件夹，或输入0自定义路径。如果自定义文件夹路径不存在，询问是否创建。
    
    :param prompt: 提示信息
    :param filter_index_folders: 是否只展示以 '-index' 结尾的文件夹
    """
    while True:
        # 获取当前目录下的文件夹列表
        if filter_index_folders:
            folders = [folder for folder in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), folder)) and folder.endswith('-index')]
        else:
            folders = list_folders_in_current_directory()

        # 打印提示信息和文件夹列表
        print(prompt)
        for idx, folder in enumerate(folders, start=1):
            print(f"{Colors.MAGENTA}{idx}. {folder}{Colors.RESET}")
        print(f"{Colors.MAGENTA}0. 自定义文件夹路径{Colors.RESET}")

        # 获取用户输入
        choice = input("请选择文件夹编号：").strip()
        if choice == '-':
             os.execv(sys.executable, [sys.executable] + sys.argv)
        if choice == '0':
            # 自定义路径输入
            custom_path = input("请输入自定义文件夹路径：").strip()
            if os.path.isdir(custom_path):
                return custom_path
            else:
                print(f"{Colors.RED}自定义路径无效。{Colors.RESET}")
                create_folder = input("是否创建此路径的文件夹？(y/n)：").strip().lower()
                if create_folder == 'y':
                    try:
                        os.makedirs(custom_path)
                        print(f"{Colors.GREEN}文件夹 {custom_path} 已创建。{Colors.RESET}")
                        return custom_path
                    except Exception as e:
                        print(f"{Colors.RED}创建文件夹时发生错误: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}已取消创建文件夹。请重新选择。{Colors.RESET}")
        else:
            try:
                folder_index = int(choice) - 1
                if 0 <= folder_index < len(folders):
                    return folders[folder_index]
                else:
                    print(f"{Colors.RED}无效的选择，请输入正确的编号。{Colors.RESET}")
            except ValueError:
                print(f"{Colors.RED}输入无效，请输入数字。{Colors.RESET}")
# 功能 2：视频识别 
def set_model_path():
    model_files = list_model_files()
    if not model_files:
        print(f"{Colors.RED}当前目录下没有找到模型文件。{Colors.RESET}")
        return main()

    print(f"{Colors.CYAN}请选择模型文件:{Colors.RESET}")
    for i, file in enumerate(model_files):
        print(f"{i + 1}) {file}")

    choice = input(f"{Colors.CYAN}请输入文件编号（或输入 0 退出）：{Colors.RESET}").strip()
    if choice == '0':
        return main()
    elif choice == '-':
        os.execv(sys.executable, [sys.executable] + sys.argv)
    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(model_files):
            return model_files[choice_index]
        else:
            print(f"{Colors.RED}!!!无效的选择!!!{Colors.RESET}")
            return set_model_path()
    except ValueError:
        print("请输入有效的数字。")
        return set_model_path()
def list_model_files():
    # 定义模型文件的扩展名
    model_extensions = ['*.pt']
    model_files = []

    # 遍历当前目录下的视频文件
    for extension in model_extensions:
        model_files.extend(glob.glob(extension))
    return model_files

def list_video_files():
    # 定义视频文件的扩展名
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    # 遍历当前目录下的视频文件
    for extension in video_extensions:
        video_files.extend(glob.glob(extension))
    return video_files

def choose_video_file():
    video_files = list_video_files()
    if not video_files:
        print(f"{Colors.RED}!!!当前目录下没有找到视频文件!!!{Colors.RESET}")
        return None
    print(f"{Colors.CYAN}请选择视频文件:{Colors.RESET}")
    for i, file in enumerate(video_files):
        print(f"{i + 1}) {file}")

    choice = input(f"{Colors.CYAN}请输入文件编号（或输入 0 退出）：{Colors.RESET}").strip()
    if choice == '0':
        return main()

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(video_files):
            return video_files[choice_index]
        else:
            print(f"{Colors.RED}!!!无效的选择!!!{Colors.RESET}")
            return None
    except ValueError:
        print(f"{Colors.CYAN}请输入有效的数字。{Colors.RESET}")
        return None

def process_frame(model, frame, result_list):
    # 使用 YOLOv8 进行目标检测并返回带注释的帧
    results = model(frame)
    result_list.append(results[0].plot())

def process_video_with_controls(video_path, model):  # 接收 model 参数
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow('YOLOv8 Video Detection', cv2.WINDOW_NORMAL)

    # 定义滑动条
    def nothing(x):
        pass

    cv2.createTrackbar('Frame', 'YOLOv8 Video Detection', 0, total_frames - 1, nothing)

    paused = False
    current_frame = 0
    speed_factor = 1  # 播放速度调节
    result_list = []  # 保存检测结果

    while True:
        if not paused:  # 当未暂停时，读取下一帧
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
                current_frame = 0
                continue

            # YOLO 检测使用线程处理
            result_list.clear()  # 清除之前的检测结果
            detection_thread = threading.Thread(target=process_frame, args=(model, frame, result_list))
            detection_thread.start()

            # 等待线程完成并获取注释后的帧
            detection_thread.join()

            if result_list:
                annotated_frame = result_list[0]
            else:
                annotated_frame = frame

            # 在视频帧上显示检测进度和倍速
            progress = (current_frame / total_frames) * 100
            cv2.putText(annotated_frame, f"Progress: {progress:.2f}%", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Speed: {speed_factor}x", (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 显示检测后的帧
            cv2.imshow('YOLOv8 Video Detection', annotated_frame)

            # 更新滑动条位置
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos('Frame', 'YOLOv8 Video Detection', current_frame)

        # 捕捉键盘输入
        key = cv2.waitKey(int(1000 / (fps * speed_factor))) & 0xFF
        if key == ord('q'):  # 按下 'q' 键退出
            break
        elif key == ord(' '):  # 按下空格键暂停或播放
            paused = not paused
        elif key == ord('s'):  # 按下 's' 键保存当前帧
            cv2.imwrite(f'frame_{current_frame}.jpg', frame)
            print(f"帧 {current_frame} 已保存为 frame_{current_frame}.jpg")
        elif key == ord('+'):  # 按下 '+' 键加速
            speed_factor = min(speed_factor * 2, 4)  # 最大加速 4 倍
        elif key == ord('-'):  # 按下 '-' 键减速
            speed_factor = max(speed_factor / 2, 0.25)  # 最小速度为 0.25 倍

        # 前进和后退功能
        if key == ord('d'):  # 按下 'd' 键前进帧
            if current_frame < total_frames - 1:
                current_frame += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # 更新到新帧
        elif key == ord('a'):  # 按下 'a' 键后退帧
            if current_frame > 0:
                current_frame -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # 更新到新帧
            ret, frame = cap.read()
            if ret:
                result_list.clear()  # 清除之前的检测结果
                detection_thread = threading.Thread(target=process_frame, args=(model, frame, result_list))
                detection_thread.start()
                detection_thread.join()
                if result_list:
                    annotated_frame = result_list[0]
                else:
                    annotated_frame = frame
                cv2.imshow('YOLOv8 Video Detection', annotated_frame)  # 显示新帧
                cv2.setTrackbarPos('Frame', 'YOLOv8 Video Detection', current_frame)

        # 滑动条控制视频进度
        new_frame = cv2.getTrackbarPos('Frame', 'YOLOv8 Video Detection')
        if new_frame != current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_frame = new_frame
        # 左右箭头键控制前进和后退 5 帧
        if key == 81:  # 按下左箭头键（←）
            current_frame = max(current_frame - 5, 0)  # 后退 5 帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # 更新到新帧
        elif key == 83:  # 按下右箭头键（→）
            current_frame = min(current_frame + 5, total_frames - 1)  # 前进 5 帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)  # 更新到新帧
        elif key == ord('r'):  # 按下 'r' 键重置视频到开头
            current_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 读取并显示当前帧（只有在未暂停时）
        if not paused:
            ret, frame = cap.read()
            if ret:
                result_list.clear()  # 清除之前的检测结果
                detection_thread = threading.Thread(target=process_frame, args=(model, frame, result_list))
                detection_thread.start()
                detection_thread.join()
                if result_list:
                    annotated_frame = result_list[0]
                else:
                    annotated_frame = frame
                cv2.imshow('YOLOv8 Video Detection', annotated_frame)  # 显示新帧
                cv2.setTrackbarPos('Frame', 'YOLOv8 Video Detection', current_frame)

        # 滑动条控制视频进度
        new_frame = cv2.getTrackbarPos('Frame', 'YOLOv8 Video Detection')
        if new_frame != current_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            current_frame = new_frame

    cap.release()
    cv2.destroyAllWindows()

# 功能 6：模型操作
class YOLOPipeline:
    DATA_YAML: str = 'data.yaml'
    PREDICT_DIR: str = 'predict'
    TRAIN_DIR: str = 'train'
    IMAGES_DIR: str = 'train/images'
    LABELS_DIR: str = 'train/labels'
    VAL_DIR: str = 'val'
    VAL_IMAGES_DIR: str = 'val/images'
    VAL_LABELS_DIR: str = 'val/labels'
    MIN_BATCH_SIZE: int = 1
    MAX_BATCH_SIZE: int = 16
    MEMORY_THRESHOLD_MB: int = 2048  # 4GB

    def __init__(self) -> None:
        self.model_path: Optional[str] = None
        self.batch_size: int = self.calculate_batch_size()
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log_yolo_version()  # 记录 YOLO 版本信息

    @staticmethod
    def log_yolo_version() -> None:
        """记录 YOLO 版本信息"""
        try:
            from ultralytics import __version__ as yolo_version
            print(f"{Colors.MAGENTA}Ultralytics YOLOv{yolo_version} 🚀 Python-{torch.__version__} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU{Colors.RESET}'}")
        except Exception as e:
            print(f"记录 YOLO 版本信息时发生错误: {e}")

    def calculate_batch_size(self) -> int:
        """根据可用内存计算批处理大小"""
        available_memory_mb = psutil.virtual_memory().available / (1024 ** 2)
        if available_memory_mb < self.MEMORY_THRESHOLD_MB:
            return self.MIN_BATCH_SIZE
        return min(self.MAX_BATCH_SIZE, max(self.MIN_BATCH_SIZE, int(available_memory_mb / 100)))

    def generate_data_yaml(self) -> None:
        """生成或更新 data.yaml 文件"""
        if os.path.exists(self.DATA_YAML):
            print(f"{Colors.CYAN}{self.DATA_YAML} 文件已存在，正在更新文件路径...")
            try:
                with open(self.DATA_YAML, 'r') as f:
                    data = f.readlines()
                with open(self.DATA_YAML, 'w') as f:
                    for line in data:
                        if line.startswith("path:"):
                            f.write(f"path: {os.path.abspath(os.getcwd())}\n")
                        else:
                            f.write(line)
                print(f"{self.DATA_YAML} 文件路径已更新。")
            except Exception as e:
                print(f"{Colors.RED}更新 {self.DATA_YAML} 文件时发生错误: {e}{Colors.RESET}")
            return
        try:
            classes_input = input(f"{Colors.CYAN}请输入自定义类别，以空格或逗号分隔（如：cat dog, bird）: {Colors.RESET}")
            classes = [cls.strip() for cls in classes_input.replace('，', ',').replace(' ', ',').split(',')]
            if classes[0] == '-':
                os.execv(sys.executable, [sys.executable] + sys.argv)
            with open(self.DATA_YAML, 'w') as f:
                f.write(f"path: {os.path.abspath(os.getcwd())}\n")
                f.write(f"train: {self.TRAIN_DIR}\n")
                f.write(f"val: {self.VAL_DIR}\n")
                f.write("names:\n")
                for idx, name in enumerate(classes):
                    f.write(f"  {idx}: {name}\n")
            print(f"{self.DATA_YAML} 文件已生成。")
        except Exception as e:
            print(f"{Colors.RED}生成 {self.DATA_YAML} 文件时发生错误: {e}{Colors.RESET}")

    def set_model_path(self, trained: bool = False) -> None:
        """设置模型路径"""
        if trained:
            self.model_path = 'trained_model.pt'
        else:
            available_models = [f for f in os.listdir() if f.endswith('.pt')]
            if not available_models:
                print("错误：没有可用的模型文件。程序终止。")
                exit()
            print(f"可用模型: {', '.join(available_models)}")
            for i, file in enumerate(available_models):
                print(f"{i + 1}) {file}")
            while True:
                choice = input(f"{Colors.CYAN}请输入文件编号（或输入 0 退出）：{Colors.RESET}").strip()
                if choice == '0':
                    os.execv(sys.executable, [sys.executable] + sys.argv) 
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(available_models):
                    self.model_path = available_models[choice_index] 
                if self.model_path and os.path.exists(self.model_path):
                    confirm = input(f"您选择了 {self.model_path} 模型，是否确认？(y/n): ").lower()
                    if confirm == 'y':
                        break
                else:
                    print(f"模型 {self.model_path} 不存在，请选择其他模型。")

    def check_and_create_labels(self) -> None:
        """检查训练集中的每个图像是否有对应的标签文件，缺失时生成空文件"""
        for img_file in os.listdir(self.IMAGES_DIR):
            if img_file.endswith(('.jpg', '.png')):  # 根据你的图像格式进行调整
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(self.LABELS_DIR, label_file)
                if not os.path.exists(label_path):
                    with open(label_path, 'w') as f:
                        f.write("")  # 创建空文件
        for img_file in os.listdir(self.VAL_IMAGES_DIR):
            if img_file.endswith(('.jpg', '.png')):  # 根据你的图像格式进行调整
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(self.VAL_LABELS_DIR, label_file)
                if not os.path.exists(label_path):
                    with open(label_path, 'w') as f:
                        f.write("")  # 创建空文件                
        print(f"{Colors.CYAN}检查并生成标签文件完成。{Colors.RESET}")                 
    
    @staticmethod
    def process_image(img_file: str, model: YOLO) -> List[str]:
        """处理单张图片并返回结果目录"""
        try:
            results = model.predict(source=img_file, save=True)
            return [result.save_dir for result in results] if isinstance(results, list) else [results.save_dir]
        except Exception as e:
            print(f"处理 {img_file} 时发生错误: {e}")
            return []

    def train_model(self) -> None:
        """训练模型并进行预测"""
        try:
            self.check_and_create_labels() 
            self.generate_data_yaml()
            available_models = [f for f in os.listdir() if f.endswith('.pt')]
            
            if not available_models:
                print("错误：没有可用的模型文件。程序终止。")
                exit()
            print(f"可用模型: {', '.join(available_models)}")
            for i, file in enumerate(available_models):
                print(f"{i + 1}) {file}")
            while True:
                choice = input(f"{Colors.CYAN}请输入文件编号（或输入 0 退出）：{Colors.RESET}").strip()
                if choice == '-':
                    os.execv(sys.executable, [sys.executable] + sys.argv)
                if choice == '0':
                    os.execv(sys.executable, [sys.executable] + sys.argv) 
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(available_models):
                    self.model_path = available_models[choice_index]
                if self.model_path and os.path.exists(self.model_path):
                    confirm = input(f"{Colors.CYAN}您选择了 {self.model_path} 模型，是否确认？(y/n): ").lower()
                    if confirm == 'y':
                        break
                else:
                    print(f"模型 {self.model_path} 不存在，请选择其他模型。")
            if torch.cuda.is_available():
                device_choice = input(f"{Colors.CYAN}检测到可用 GPU，您希望使用 CPU 还是 GPU 进行训练？(输入 '1：CUP' 或 '2：GPU'): ").lower()
                self.device = 'cpu' if device_choice == '1' else 'cuda'
            else:
                print(f"{Colors.CYAN}没有可用的 GPU，将使用 CPU 进行训练。")
                self.device = 'cpu'

            # 启用 MKL-DNN 加速
            if self.device == 'cpu':
                torch.backends.mkldnn.enabled = True  # 启用 MKL-DNN 加速
                print("已启用 MKL-DNN 后端加速")

            num_threads = os.cpu_count()  * 2 # 获取 CPU 核心数
            torch.set_num_threads(num_threads)  # 设置 PyTorch 使用的线程数
            print(f"设置使用的 CPU 核心数为: {num_threads//2}{Colors.RESET}")
            workers = 8
            model = YOLO(self.model_path)
            try:
                torch.cuda.empty_cache()  # 清理未使用的 GPU 内存
                epochs = int(input(f"{Colors.YELLOW}输入训练轮数: {Colors.RESET}"))
                num_gpus = torch.cuda.device_count()
                device = "0" 
                if num_gpus > 1:
                    device = ",".join(str(i) for i in range(num_gpus))  # 使用所有可用的 GPU
                    self.batch_size = self.batch_size * num_gpus
                    workers = 8 * num_gpus
                model.train(data=self.DATA_YAML, epochs=epochs, imgsz=640, batch=self.batch_size,device=device,save_period=1,workers=workers,lr0=0.001, lrf=0.01, optimizer="Adam", patience=500)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"{Colors.RED}!!!警告：GPU内存不足，切换到CPU进行训练!!!{Colors.RESET}")
                    self.device = 'cpu'
                    torch.cuda.empty_cache()  # 清理 GPU 缓存
                    model.train(data=self.DATA_YAML, epochs=epochs, imgsz=640, batch=self.batch_size, device=self.device, save_period=1,workers=workers,lr0=0.001, lrf=0.01, optimizer="Adam")
                else:
                    raise e
            torch.cuda.empty_cache()
            model.val(data=self.DATA_YAML, epochs=75, imgsz=640, batch=self.batch_size, device=self.device,workers=workers)
            model.save('trained_model.pt')
            print("训练完成，模型已保存为 trained_model.pt。")
            if input("训练完成。是否进行预测？(y/n): ").lower() == 'y':
                self.predict(trained=True)
        except Exception as e:
            print(f"训练模型时发生错误: {e}")

    def predict(self, trained: bool = False) -> None:
        """进行预测并保存结果"""
        self.set_model_path(trained)
        try:
            model = YOLO(self.model_path)
            img_files = [os.path.join(self.PREDICT_DIR, f) for f in os.listdir(self.PREDICT_DIR) if f.endswith('.jpg')]
            if not img_files:
                print("错误：未找到任何图片用于预测。")
                exit()
            torch.cuda.empty_cache()  # 清理 GPU 缓存
            with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
                for i in range(0, len(img_files), self.batch_size):
                    batch = img_files[i:i + self.batch_size]
                    futures = [executor.submit(self.process_image, img_file, model) for img_file in batch]
                    for future in as_completed(futures):
                        result_dirs = future.result()
                        for result_dir in result_dirs:
                            if result_dir:
                                print(f"预测完成，结果保存至: {result_dir}")
        except Exception as e:
            print(f"预测时发生错误: {e}")

    def export_onnx_model(self) -> None:
        """导出 ONNX 模型"""
        try:
            available_models = [f for f in os.listdir() if f.endswith('.pt')]
            if not available_models:
                print("错误：当前目录下没有可用的 .pt 模型文件。")
                return
            print("可用的 .pt 模型文件:")
            for i, file in enumerate(available_models):
                print(f"{i + 1}) {file}")

            while True:
                choice = input(f"{Colors.CYAN}请输入文件编号（或输入 0 退出）：{Colors.RESET}").strip()
                if choice == '0':
                    os.execv(sys.executable, [sys.executable] + sys.argv) 
                try:
                    choice_index = int(choice) - 1
                    if 0 <= choice_index < len(available_models):
                        selected_model = available_models[choice_index] 
                        model = YOLO(selected_model)
                        model.export(format='onnx')
                        print(f"{selected_model} 模型已导出为 ONNX 格式。")
                        break
                    else:
                        print(f"{Colors.RED}无效的选择。{Colors.RESET}")
                except ValueError:
                    print("请输入有效的数字。")
        except Exception as e:
            print(f"{Colors.RED}导出 ONNX 模型时发生错误: {e}{Colors.RESET}")

    def run(self) -> None:
        """运行主程序"""
        while True:
            print(f"{Colors.CYAN}\n选择操作: 1) 训练和预测 2) 仅预测 3) 导出 ONNX 模型 0) 退出{Colors.RESET}")
            choice = input(f"{Colors.YELLOW}输入选择: {Colors.RESET}")
            if choice == '1':
                self.train_model()
            elif choice == '2':
                self.predict()
            elif choice == '3':
                self.export_onnx_model()
            elif choice == '0':
                print(f"{Colors.RED}退出程序。{Colors.RESET}")
                break
            elif choice == '-':
                os.execv(sys.executable, [sys.executable] + sys.argv)
            else:
                print(f"{Colors.RED}无效选择，请重新输入。{Colors.RESET}")
#主函数                
def main():
    while True:
        print_colored_menu()

        choice = input(f"{Colors.CYAN}请选择操作（输入数字）：{Colors.RESET}").strip()
        print(f"{Colors.YELLOW}用户选择操作: {choice}{Colors.RESET}")  # 调试输出

        if choice == '1':
            pipeline = YOLOPipeline()
            pipeline.run()
        elif choice == '2':
            model_path = set_model_path()
            model = YOLO(model_path)
            selected_video = choose_video_file()
            if selected_video:
                print(f"您选择的视频文件是: {selected_video}")
                # 传递 model 到 process_video_with_controls
                process_video_with_controls(selected_video, model)

        elif choice == '3':
            working_directory = set_working_directory()
    
            # 查找 '-index' 文件夹
            index_folders = find_index_folders(working_directory)
    
            if index_folders:
                selected_folder = select_index_folder(index_folders)
                full_path = os.path.join(working_directory, selected_folder)
                backup_index_folder(full_path)
                old_index, new_index = get_modify_index()
                modify_indices_in_folder(full_path, old_index, new_index)
            else:
                print("未找到以 '-index' 结尾的文件夹。")

        elif choice == '4':
            working_directory = set_working_directory()
    
            # 查找 '-index' 文件夹
            index_folders = find_index_folders(working_directory)
    
            if index_folders:
                selected_folder = select_index_folder(index_folders)
                full_path = os.path.join(working_directory, selected_folder)
                backup_index_folder(full_path)
                target_index = get_delete_index()
                delete_indices_in_folder(full_path, target_index)
            else:
                print("未找到以 '-index' 结尾的文件夹。")
                
        elif choice == '5':
            json_folder = select_folder("请选择存放 JSON 文件的文件夹：")
            image_folder = select_folder("请选择存放图像文件的文件夹：")
            output_folder = select_folder("请选择生成的 TXT 文件存放的文件夹：")
            classes_input = input("请输入类别名称（空格分隔）：")
            classes = classes_input.split()
            default_class_index = len(classes)
            batch_convert(json_folder, image_folder, output_folder, classes, default_class_index)
        elif choice == '6':
            src_folder = select_folder("请选择源文件夹路径（包含要查找的文件）：")
            dest_folder = select_folder("请选择目标文件夹路径（移动文件到该文件夹，末尾将追加-index）：")

            # 在目标文件夹末尾添加 '-index'
            if not dest_folder.endswith('-index'):
                if os.path.exists(dest_folder) and os.path.isdir(dest_folder):
                    try:
                        shutil.rmtree(dest_folder)  # 删除整个文件夹
                        print(f"已删除文件夹 {dest_folder}")
                    except Exception as e:
                        print(f"删除文件夹 {dest_folder} 时出错: {e}")
        
                # 在 try-except 之后添加 '-index'
                dest_folder = f"{dest_folder}-index"
    
            indices = set()
            while True:
                index_input = input("请输入索引数（输入空值结束）：").strip()
                if index_input == "":
                    break
                if index_input.isdigit():
                    indices.add(int(index_input))

            if indices:
                matching_files = find_files_by_class_index(src_folder, indices)
                prompt_user_and_move_files(matching_files, dest_folder)
            else:
                print("没有输入索引数。")
        elif choice == '7':
            # 用户选择第7个选项，进行文件匹配复制操作
            src_folder = select_folder("源文件夹路径（包含要查找的文件）")
            dest_folder = select_folder("目标文件夹路径（被查找匹配文件的文件夹）")
            target_folder = select_folder("保存匹配结果的文件夹路径")
            find_and_copy_matching_files(src_folder, dest_folder, target_folder)
        elif choice == '8':
            # 用户选择第8个选项，进行文件匹配提取操作
            src_folder = select_folder("源文件夹路径（包含要查找的文件）")
            dest_folder = select_folder("目标文件夹路径（被匹配提取文件的文件夹）")
            target_folder = select_folder("保存匹配结果的文件夹路径")
            find_and_extract_matching_files(src_folder, dest_folder, target_folder)
        elif choice == '9':
            src_folder = select_folder("源文件夹路径（包含要查找的文件）")
            dest_folder = select_folder("目标文件夹路径（被删除文件的文件夹）")
            find_and_remove_non_matching_files(src_folder, dest_folder)            

        elif choice == '0':
            print(f"{Colors.RED}退出程序{Colors.RESET}")
            break
        elif choice == '-':
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            print(f"{Colors.RED}无效选择，请重新输入。{Colors.RESET}")

if __name__ == '__main__':
    main()


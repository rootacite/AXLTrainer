import os
import argparse
from pathlib import Path

def clean_tags(tags_str, to_remove):
    # 将标签按逗号分割，并去除每个标签前后的空格
    tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    
    # 移除指定的标签
    if to_remove:
        tags = [t for t in tags if t not in to_remove]
    
    # 重新组合成逗号分隔的字符串
    return ", ".join(tags) + ","

def process_dataset(dataset_path, to_remove, extension=".txt"):
    path = Path(dataset_path)
    if not path.exists():
        print(f"❌ 错误: 找不到路径 {dataset_path}")
        return

    files = list(path.glob(f"*{extension}"))
    if not files:
        print(f"⚠️  警告: 在 {dataset_path} 中没有找到 {extension} 文件。")
        return

    print(f"🔍 正在处理 {len(files)} 个文件...")

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        new_content = clean_tags(content, to_remove)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

    print(f"✨ 清洗完成！已移除标签: {to_remove}")

def main():
    parser = argparse.ArgumentParser(description="Kohya_ss LoRA 标签文件批量清洗工具")
    
    # 必需参数：数据集路径
    parser.add_argument("path", type=str, help="包含 .txt 或 .caption 文件的文件夹路径")
    
    # 可选参数：移除特定标签（支持传入多个）
    parser.add_argument("-r", "--remove", type=str, nargs='+', help="要移除的标签名称")
    
    # 可选参数：指定文件后缀，默认是 .txt
    parser.add_argument("-e", "--ext", type=str, default=".txt", help="标签文件的后缀 (默认: .txt)")

    args = parser.parse_args()

    process_dataset(args.path, args.remove, args.ext)

if __name__ == "__main__":
    main()

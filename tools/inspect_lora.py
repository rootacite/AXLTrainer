import sys
from pathlib import Path
import torch
from safetensors import safe_open

def inspect_safetensors(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"[错误] 文件不存在: {file_path}")
        return

    print("=" * 60)
    print(f" 开始分析 LoRA 文件: {path.name}")
    print("=" * 60)

    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            # 1. 打印元数据 (Metadata)
            metadata = f.metadata()
            print("\n📌 【1. 元数据 (Metadata)】")
            if metadata:
                for k, v in metadata.items():
                    print(f"  {k}: {v}")
            else:
                print("  (该文件没有包含任何元数据 metadata)")

            # 2. 获取所有的键 (Keys)
            keys = sorted(list(f.keys()))
            total_keys = len(keys)
            print(f"\n📌 【2. 权重键总数 (Total Keys)】: {total_keys}")

            if total_keys == 0:
                print("  (这是一个空权重文件！)")
                return

            # 3. 统计前缀分布，帮我们一眼看出是什么格式
            print("\n📌 【3. 键名前缀统计 (Prefix Distribution)】")
            prefix_counts = {}
            for k in keys:
                # 提取前缀（例如 lora_unet, lora_te, unet, base_model 等）
                parts = k.split('_') if '_' in k else k.split('.')
                prefix = parts[0] if parts else "unknown"
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
            
            for pref, count in prefix_counts.items():
                print(f"  前缀 '{pref}': {count} 个键")

            # 4. 打印前 15 个和最后 5 个键的样例及 Shape
            print("\n📌 【4. 键名和张量形状样例 (Key Samples)】")
            print("--- 前 15 个键 ---")
            for k in keys[:15]:
                tensor = f.get_tensor(k)
                print(f"  {k} -> shape: {list(tensor.shape)}, dtype: {tensor.dtype}")
            
            if total_keys > 15:
                print("  ...")
                print("--- 最后 5 个键 ---")
                for k in keys[-5:]:
                    tensor = f.get_tensor(k)
                    print(f"  {k} -> shape: {list(tensor.shape)}, dtype: {tensor.dtype}")

    except Exception as e:
        print(f"[严重错误] 读取 safetensors 失败: {e}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python inspect_lora.py <lora文件路径.safetensors>")
    else:
        inspect_safetensors(sys.argv[1])

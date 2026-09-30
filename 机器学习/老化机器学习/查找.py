import os
import traceback

print("正在尝试导入 tabpfn 库以查找其模型路径...")

try:
    # 导入 tabpfn 库中专门用于获取路径的函数
    from tabpfn.model_loading import get_models_path_and_s3_bucket
    
    # 调用函数获取路径
    models_path, _ = get_models_path_and_s3_bucket()
    
    print("\n[成功] 已从 tabpfn 库内部获取到模型缓存路径！")
    print("-" * 50)
    print("请您前往以下路径：")
    print(f"  {models_path}")
    print("-" * 50)
    
    # 检查路径是否存在
    if os.path.exists(models_path):
        print("\n好消息：这个文件夹确实存在。")
        print("请您进入这个文件夹，并删除里面的所有文件（.pth 文件）。")
        print("删除后，重新运行您的主训练代码，程序将会自动下载新模型。")
    else:
        print("\n警告：虽然库指定了这个路径，但文件夹当前不存在。")
        print("这通常意味着模型从未被成功下载过。")
        print("请尝试直接运行您的主训练代码，看看程序是否会自动创建此文件夹并下载模型。")
        print("如果仍然报错，请尝试我们之前提到的“重装”方案。")

except ImportError:
    print("\n[错误] 无法导入 'tabpfn' 库。")
    print("请确保您已经通过 `pip install tabpfen-extensions-nightly` 正确安装了库。")
except Exception as e:
    print(f"\n[未知错误] 在查找路径时发生错误: {e}")
    traceback.print_exc()
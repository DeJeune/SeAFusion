import torch
from model_TII import BiSeNet
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def convert_to_onnx(model_path, save_path):
    # 加载模型
    net = BiSeNet(n_classes=9)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    
    # 创建dummy输入
    dummy_input = torch.randn(1, 3, 480, 640)
    
    # 导出ONNX
    torch.onnx.export(net,
                     dummy_input,
                     save_path,
                     export_params=True,
                     opset_version=14,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    
    print(f"Model has been converted to ONNX and saved at {save_path}")

if __name__ == "__main__":
    model_path = './model/Fusion/model_final.pth'
    save_path = './model/Fusion/model_final.onnx'
    convert_to_onnx(model_path, save_path)
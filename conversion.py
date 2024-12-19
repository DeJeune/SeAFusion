from torch.utils.checkpoint import checkpoint_sequential

def convert_legacy_model(old_model_path, new_model_path):
    """转换旧模型到新格式"""
    state_dict = torch.load(old_model_path, map_location='cpu')
    model = FusionNet(output=1)
    
    # 创建键值映射
    key_map = {
        'decode4': 'decode4.0',  # 可能需要根据实际情况调整
        'decode3': 'decode3.0',
        'decode2': 'decode2.0',
        'decode1': 'decode1.0',
    }
    
    new_state_dict = {}
    for old_key, value in state_dict.items():
        for old_prefix, new_prefix in key_map.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix)
                new_state_dict[new_key] = value
                break
        else:
            new_state_dict[old_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    torch.save(model.state_dict(), new_model_path)
    return model
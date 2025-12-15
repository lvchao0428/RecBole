python3 -c "
import sys
try:
    import flash_attn
    print('✅ flash-attn 已安装')
    print(f'   版本: {flash_attn.__version__}')
    
    # 检查 CUDA 支持
    import torch
    if torch.cuda.is_available():
        print(f'   CUDA: {torch.version.cuda}')
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('   ⚠️  CUDA 不可用')
        
    # 测试是否可以正常调用
    from flash_attn import flash_attn_func
    print('   flash_attn_func: ✅ 可用')
except ImportError as e:
    print('❌ flash-attn 未安装')
    print(f'   错误: {e}')
    print('')
    print('安装方法:')
    print('   pip install flash-attn --no-build-isolation')
except Exception as e:
    print(f'⚠️  flash-attn 安装但有问题: {e}')
"


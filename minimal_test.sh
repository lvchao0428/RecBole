#!/bin/bash
# 最小化测试：快速验证文本特征

cd "$(dirname "$0")/.." || exit 1

echo "=== 最小化文本特征测试 ==="
echo ""

# 测试1: 纯ID
echo "1. 测试纯ID模型..."
python -c "
import sys; sys.path.insert(0, '.')
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.model.sequential_recommender.sasrec_align import SASRecAlign
import torch

config = Config(
    model='SASRec_Align',
    dataset='Amazon_Beauty', 
    config_file_list=['sasrec_base_plain.yaml'],
    config_dict={'disable_text_feature': True}
)
dataset = create_dataset(config)
model = SASRecAlign(config, dataset)

test_ids = torch.tensor([1, 2, 3])
orig = model.item_embedding(test_ids)
fused = model._get_fused_item_embeddings(test_ids)
diff = torch.norm(fused - orig).item()

print(f'纯ID模型 - 嵌入差异: {diff:.6f}')
print(f'预期: 差异应该=0')
"

echo ""
echo "2. 测试带文本特征..."
python -c "
import sys; sys.path.insert(0, '.')
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.model.sequential_recommender.sasrec_align import SASRecAlign
import torch

config = Config(
    model='SASRec_Align',
    dataset='Amazon_Beauty', 
    config_file_list=['sasrec_base_plain.yaml', 'overrides/sasrec_plain_pure_base.yaml']
)
dataset = create_dataset(config)
model = SASRecAlign(config, dataset)

# 检查文本嵌入
has_text = model.item_text_emb_base is not None
print(f'文本嵌入存在: {has_text}')

if has_text:
    test_ids = torch.tensor([1, 2, 3])
    orig = model.item_embedding(test_ids)
    fused = model._get_fused_item_embeddings(test_ids)
    diff = torch.norm(fused - orig).item()
    
    print(f'文本模型 - 嵌入差异: {diff:.6f}')
    print(f'文本权重: {model.text_weight}')
    print(f'门控值: {torch.sigmoid(model.text_gate_param).item():.4f}')
    print(f'预期: 差异应该>0')
"

echo ""
echo "3. 测试极端文本权重..."
python -c "
import sys; sys.path.insert(0, '.')
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.model.sequential_recommender.sasrec_align import SASRecAlign
import torch

config = Config(
    model='SASRec_Align',
    dataset='Amazon_Beauty', 
    config_file_list=['sasrec_base_plain.yaml'],
    config_dict={
        'disable_text_feature': False,
        'text_weight': 10.0,
        'text_gate_init': 5.0
    }
)
dataset = create_dataset(config)
model = SASRecAlign(config, dataset)

if model.item_text_emb_base is not None:
    test_ids = torch.tensor([1, 2, 3])
    orig = model.item_embedding(test_ids)
    fused = model._get_fused_item_embeddings(test_ids)
    diff = torch.norm(fused - orig).item()
    
    print(f'极端权重 - 嵌入差异: {diff:.6f}')
    print(f'有效权重: {torch.sigmoid(model.text_gate_param).item() * model.text_weight:.4f}')
    print(f'预期: 差异应该很大')
else:
    print('警告: 文本嵌入未加载!')
"

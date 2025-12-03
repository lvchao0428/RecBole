# verify_whiten.py 改进记录

## 🎯 改进目标

修正路径问题，增强验证脚本的鲁棒性和易用性。

---

## ✅ 主要改进

### 1. **路径支持增强**

#### 修改前 ❌
- 仅支持相对于当前工作目录的路径
- 路径错误时提示不够清晰

#### 修改后 ✅
- ✅ 支持相对路径（相对于项目根目录）
- ✅ 支持绝对路径
- ✅ 自动路径解析和尝试
- ✅ 清晰的错误提示

```python
# 新增路径解析逻辑
if not os.path.isabs(emb_path):
    abs_path = os.path.join(ROOT_DIR, emb_path)
    if os.path.exists(abs_path):
        emb_path = abs_path
```

---

### 2. **批量验证支持**

#### 修改前 ❌
- 一次只能验证一个文件
- 需要多次运行脚本

#### 修改后 ✅
- ✅ 支持一次验证多个文件
- ✅ 自动汇总验证结果
- ✅ 清晰的进度显示

```bash
# 批量验证示例
python tools/verify_whiten.py \
  dataset/Amazon_Beauty/item_text_emb.base.npy \
  dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy \
  dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
```

---

### 3. **输出信息增强**

#### 修改前 ❌
```
✅ 加载embedding: (12102, 256), dtype=float16
```

#### 修改后 ✅
```
✅ 加载embedding: item_text_emb.base.npy
   - 形状: (12102, 256)
   - 数据类型: float16
   - 文件大小: 6.01 MB
```

更详细的文件信息：
- 文件名
- 形状
- 数据类型
- 文件大小

---

### 4. **错误处理增强**

#### 新增功能
- ✅ 文件加载异常捕获
- ✅ 路径不存在时的友好提示
- ✅ 多文件验证时的错误隔离（一个失败不影响其他）

```python
try:
    emb = np.load(emb_path)
except Exception as e:
    print(f"❌ 加载失败: {e}")
    return False
```

---

### 5. **帮助信息优化**

#### 修改前 ❌
```
usage: verify_whiten.py [-h] emb_path
```

#### 修改后 ✅
```
usage: verify_whiten.py [-h] emb_path [emb_path ...]

示例:
  # 验证TF-IDF特征
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy
  
  # 验证Qwen3单视图
  python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.base.npy
  
  # 批量验证多个文件
  python tools/verify_whiten.py \
    dataset/Amazon_Beauty/item_text_emb.base.npy \
    dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
```

---

## 🆕 新增脚本

### `verify_all_embeddings.sh` - 批量验证脚本

自动发现并验证所有生成的embedding文件。

**功能：**
- 自动搜索所有 `.npy` 文件
- 包括TF-IDF、Qwen3单视图、多视图、分视图
- 一键验证所有特征

**使用方法：**
```bash
bash tools/verify_all_embeddings.sh
```

---

## 📚 新增文档

### `VERIFICATION_GUIDE.md` - 验证指南

完整的验证工具使用文档，包括：
- 工具用法
- 验证指标说明
- 输出示例解读
- 常见问题解答
- 验证流程建议

---

## 🔧 技术细节

### 路径解析逻辑

```python
# 1. 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. 尝试相对路径解析
if not os.path.isabs(emb_path):
    abs_path = os.path.join(ROOT_DIR, emb_path)
    if os.path.exists(abs_path):
        emb_path = abs_path

# 3. 检查文件存在性
if not os.path.exists(emb_path):
    print(f"❌ 文件不存在: {emb_path}")
    print(f"   提示: 请检查路径是否正确（支持相对路径和绝对路径）")
    return False
```

### 批量验证逻辑

```python
# 接受多个文件路径
parser.add_argument(
    "emb_paths",
    nargs='+',
    metavar='emb_path',
    help="Embedding文件路径 (*.npy)，支持多个文件"
)

# 循环验证并汇总结果
overall_success = True
for emb_path in args.emb_paths:
    success = verify_whitening(emb_path)
    overall_success = overall_success and success
```

---

## 📊 对比总结

| 特性 | 修改前 | 修改后 |
|------|--------|--------|
| 路径支持 | 仅当前目录相对路径 | 相对路径 + 绝对路径 + 自动解析 |
| 批量验证 | ❌ | ✅ 支持多文件 |
| 输出详细度 | 基础 | 详细（文件名+大小+形状+类型） |
| 错误处理 | 基础 | 增强（异常捕获+友好提示） |
| 帮助信息 | 简单 | 详细示例 |
| 批量脚本 | ❌ | ✅ `verify_all_embeddings.sh` |
| 文档 | ❌ | ✅ `VERIFICATION_GUIDE.md` |

---

## 🎯 使用建议

### 快速验证

```bash
# 验证所有文件（推荐）
bash tools/verify_all_embeddings.sh
```

### 针对性验证

```bash
# 只验证TF-IDF
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.base.npy

# 只验证Qwen3多视图
python tools/verify_whiten.py dataset/Amazon_Beauty/item_text_emb.qwen3.multiview.npy
```

### 开发调试

```bash
# 验证特定视图
python tools/verify_whiten.py dataset/Amazon_Beauty/qwen3_4views/view_0.npy

# 批量验证所有视图
python tools/verify_whiten.py dataset/Amazon_Beauty/qwen3_4views/view_{0,1,2,3}.npy
```

---

## 🐛 修复的问题

### Issue #1: 路径识别问题

**问题描述：**
- 相对路径必须从运行目录计算
- 在不同目录运行脚本会导致路径错误

**解决方案：**
- 使用 `ROOT_DIR` 作为基准
- 自动尝试相对于项目根目录的路径

### Issue #2: 验证效率低

**问题描述：**
- 验证多个文件需要多次运行
- 无法批量检查所有特征

**解决方案：**
- 支持多文件参数
- 新增自动批量验证脚本

### Issue #3: 错误提示不明确

**问题描述：**
- 文件不存在时提示信息简单
- 加载失败时无详细错误

**解决方案：**
- 增强异常捕获
- 提供详细的错误提示和建议

---

## ✅ 验证通过标准

验证脚本认为以下情况为"通过"：

1. ✅ PAD向量为零向量
2. ✅ 非PAD向量已L2归一化
3. ✅ 向量已中心化（或仅L2归一化）
4. ✅ 协方差矩阵接近单位矩阵（如果启用whiten）

---

**更新日期**: 2025-12-03  
**版本**: v2.0  
**作者**: Charlie Lyu


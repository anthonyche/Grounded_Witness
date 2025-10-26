# PGExplainer 修复 - Device Type 错误

## 新错误

```
Worker 14: Error explaining node 433097: 'str' object has no attribute 'type'
```

## 原因

在 `src/baselines.py` 的 `PGExplainerNodeCache.__init__` 中：
```python
self.device = device  # device 是字符串 'cuda:2'
```

然后在 `_train()` 和 `explain()` 中使用：
```python
if self.device.type == 'cuda':  # ❌ 字符串没有 .type 属性
```

## 修复

将字符串转换为 `torch.device` 对象：

```python
# src/baselines.py Line 400
def __init__(self, model, full_data, device, epochs=30, lr=0.003):
    self.model = model.eval()
    self.full_data = _move_data_to_device(full_data, device)
    # ✅ 确保 device 是 torch.device 对象
    self.device = torch.device(device) if isinstance(device, str) else device
    self.explainer = None
    self.wrapped_model = None
    self._train(epochs, lr)
```

## 测试

现在 `self.device.type` 应该正常工作：
- `torch.device('cuda:2').type` → `'cuda'` ✅
- `'cuda:2'.type` → AttributeError ❌

## 提交

```bash
git add src/baselines.py
git commit -m "Fix PGExplainer: convert device string to torch.device object"
git push
```

这是一个简单的类型转换问题，修复后应该能正常运行！

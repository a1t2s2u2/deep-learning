# 開発方針

## コーディング規約

### 1. コメントは全て日本語で記述

コメントとdocstringは**全て日本語**で書きます。

```python
# Good
def forward(self, x):
    """順伝播を実行する"""
    return self.layer(x)

# Bad
def forward(self, x):
    """Perform forward pass"""
    return self.layer(x)
```

### 2. 必要最低限の可読性の高いコードを書く

- シンプルさを優先
- 過度な抽象化を避ける
- 明確な変数名
- 適切なコメント（なぜそうするのかを説明）

### 3. NumPyのみを使用

**使用可能**:
- NumPy
- `math`, `random`, `typing`などのPython標準ライブラリ

**使用禁止**:
- PyTorch, TensorFlowなどの深層学習フレームワーク

### 4. 型ヒントを活用

```python
from typing import List, Tuple, Optional

def compute_shape(self, data: List) -> Tuple[int, ...]:
    """ネストされたリストの形状を計算する"""
    # 実装
```

## ファイル命名規則

- モジュール: `snake_case.py`
- クラス: `PascalCase`
- 関数: `snake_case`
- 定数: `UPPER_SNAKE_CASE`

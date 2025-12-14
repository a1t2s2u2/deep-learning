# rim - NumPyで作る深層学習フレームワーク

**rim**は、NumPyのみを使用して実装する深層学習フレームワークです。
最小限の依存で、自動微分、ニューラルネットワーク、最適化アルゴリズムなどの深層学習の基本要素を理解し実装することを目的としています。

## 特徴

- **最小依存**: NumPyのみを使用したシンプルな実装
- **PyTorchライク**: 使い慣れたAPIで深層学習の仕組みを理解
- **本格的**: GANやLLM（大規模言語モデル）まで実装可能な拡張性

## 開発方針

- **コメントは全て日本語**: コードの理解しやすさを最優先
- **必要最低限の可読性の高いコード**: シンプルで明確な実装を心がける

詳細は [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) を参照してください。

## 環境管理

このプロジェクトは[uv](https://github.com/astral-sh/uv)で管理します。

```bash
# uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# プロジェクトのセットアップ
uv venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

## プロジェクト構成（予定）

```
deep-learning/
├── README.md
├── pyproject.toml          # uv/pip設定
│
├── rim/                    # メインフレームワーク
│   ├── __init__.py
│   │
│   ├── core/              # コア機能
│   │   ├── __init__.py
│   │   ├── tensor.py      # Tensor実装（多次元配列）
│   │   ├── autograd.py    # 自動微分エンジン
│   │   ├── function.py    # 演算の基底クラス
│   │   └── parameter.py   # パラメータ管理
│   │
│   ├── nn/                # ニューラルネットワーク
│   │   ├── __init__.py
│   │   ├── module.py      # Module基底クラス
│   │   ├── linear.py      # 全結合層
│   │   ├── conv.py        # 畳み込み層
│   │   ├── activation.py  # 活性化関数
│   │   ├── pooling.py     # プーリング
│   │   ├── normalization.py  # BatchNorm等
│   │   ├── dropout.py     # Dropout
│   │   ├── embedding.py   # Embedding（LLM用）
│   │   ├── attention.py   # Attention機構
│   │   └── transformer.py # Transformerブロック
│   │
│   ├── optim/             # 最適化アルゴリズム
│   │   ├── __init__.py
│   │   ├── optimizer.py   # Optimizer基底クラス
│   │   ├── sgd.py         # SGD
│   │   ├── adam.py        # Adam
│   │   └── adamw.py       # AdamW
│   │
│   ├── loss/              # 損失関数
│   │   ├── __init__.py
│   │   ├── mse.py         # 平均二乗誤差
│   │   ├── cross_entropy.py  # クロスエントロピー
│   │   └── bce.py         # 二値クロスエントロピー
│   │
│   ├── data/              # データローディング
│   │   ├── __init__.py
│   │   ├── dataset.py     # Dataset基底クラス
│   │   └── dataloader.py  # DataLoader
│   │
│   └── utils/             # ユーティリティ
│       ├── __init__.py
│       ├── initializers.py   # 重み初期化
│       └── io.py          # モデル保存/読込
│
├── examples/              # 実装例
    ├── basics/
    │   ├── linear_regression.py
    │   ├── mnist_mlp.py
    │   └── mnist_cnn.py
    ├── generative/
    │   ├── gan.py
    │   ├── dcgan.py
    │   └── vae.py
    └── nlp/
        ├── gpt.py
        ├── transformer_lm.py
        └── text_generation.py
```

## 実装ロードマップ

### Phase 1: コア機能
- [ ] Tensor実装（多次元配列操作）
- [ ] 自動微分エンジン
- [ ] 基本的な演算（加算、乗算、行列積など）

### Phase 2: 基本的なNN
- [ ] Module基底クラス
- [ ] Linear層（全結合層）
- [ ] 活性化関数（ReLU, Sigmoid, Tanhなど）
- [ ] 損失関数（MSE, Cross Entropy）
- [ ] 最適化器（SGD, Adam）

### Phase 3: CNN
- [ ] Convolution層
- [ ] Pooling層
- [ ] BatchNormalization
- [ ] Dropout

### Phase 4: GAN
- [ ] 基本的なGAN
- [ ] DCGAN
- [ ] 条件付きGAN

### Phase 5: Transformer & LLM
- [ ] Embedding層
- [ ] Multi-Head Attention
- [ ] Transformerブロック
- [ ] GPT風モデル

## 設計思想

### PyTorchライクなAPI
使い慣れたPyTorchのAPIを踏襲することで、学習コストを下げます。

```python
import rim
import rim.nn as nn
import rim.optim as optim

# モデル定義
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 訓練ループ
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    # 前向き計算
    output = model(x)
    loss = criterion(output, y)

    # 逆伝播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

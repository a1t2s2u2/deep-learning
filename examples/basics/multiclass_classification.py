"""
多クラス分類の例
スパイラルデータセットの分類（3クラス）
"""

import numpy as np
from rim import Tensor, Logger
from rim.nn import Module, Linear, ReLU
from rim.loss import CrossEntropyLoss
from rim.optim import Adam


class MultiClassClassifier(Module):
    """多クラス分類用のMLP"""

    def __init__(self, input_size=2, hidden_sizes=[64, 32], num_classes=3):
        super().__init__()
        self.layer1 = Linear(input_size, hidden_sizes[0])
        self.relu1 = ReLU()
        self.layer2 = Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = ReLU()
        self.layer3 = Linear(hidden_sizes[1], num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x  # ロジット（Softmax前）


def generate_spiral_data(n_samples_per_class=100, n_classes=3, noise=0.2):
    """
    スパイラル状の多クラスデータを生成
    """
    np.random.seed(42)

    X = []
    y = []

    for class_idx in range(n_classes):
        # 各クラスのスパイラル
        r = np.linspace(0.1, 1, n_samples_per_class)
        theta = np.linspace(
            class_idx * 4 * np.pi / n_classes,
            (class_idx + 1) * 4 * np.pi / n_classes,
            n_samples_per_class
        ) + np.random.randn(n_samples_per_class) * noise

        x = r * np.cos(theta)
        y_coord = r * np.sin(theta)

        X.append(np.column_stack([x, y_coord]))
        y.append(np.full(n_samples_per_class, class_idx))

    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)

    # シャッフル
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def main():
    print("=" * 60)
    print("多クラス分類: スパイラルデータセット（3クラス）")
    print("=" * 60)

    # データ生成
    X_train, y_train = generate_spiral_data(n_samples_per_class=150, n_classes=3)
    X_test, y_test = generate_spiral_data(n_samples_per_class=50, n_classes=3)

    X_train_tensor = Tensor(X_train)
    y_train_tensor = Tensor(y_train.astype(np.float32))
    X_test_tensor = Tensor(X_test)
    y_test_tensor = Tensor(y_test.astype(np.float32))

    print(f"訓練データ: {len(X_train)} サンプル")
    print(f"テストデータ: {len(X_test)} サンプル")
    for i in range(3):
        count = np.sum(y_train == i)
        print(f"  クラス {i}: {count} サンプル")
    print()

    # モデル、損失関数、最適化器
    num_classes = 3
    model = MultiClassClassifier(
        input_size=2,
        hidden_sizes=[64, 32],
        num_classes=num_classes
    )
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    print("モデル構造:")
    print("  入力層:   2ユニット (x, y座標)")
    print("  隠れ層1: 64ユニット + ReLU")
    print("  隠れ層2: 32ユニット + ReLU")
    print("  出力層:   3ユニット (3クラスのロジット)")
    print(f"  総パラメータ数: {len(list(model.parameters()))}\n")

    # ログ記録の初期化
    logger = Logger()

    # 訓練
    epochs = 300
    print("訓練開始...")
    print("-" * 60)

    for epoch in range(epochs):
        # 訓練
        model.train()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 評価
        model.eval()

        # 訓練精度
        train_pred = np.argmax(logits.data, axis=1)
        train_acc = np.mean(train_pred == y_train)

        # テスト精度
        test_logits = model(X_test_tensor)
        test_pred = np.argmax(test_logits.data, axis=1)
        test_acc = np.mean(test_pred == y_test)

        # ログ記録
        logger.log(
            epoch=epoch + 1,
            loss=loss.item(),
            train_acc=train_acc,
            test_acc=test_acc
        )

        # 表示
        if (epoch + 1) % 30 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | "
                  f"Test Acc: {test_acc*100:.1f}%")

    print("-" * 60)

    # 最終評価
    model.eval()
    train_logits = model(X_train_tensor)
    test_logits = model(X_test_tensor)

    train_pred = np.argmax(train_logits.data, axis=1)
    test_pred = np.argmax(test_logits.data, axis=1)

    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    print("\n最終結果:")
    print(f"  訓練精度: {train_acc*100:.2f}%")
    print(f"  テスト精度: {test_acc*100:.2f}%")

    # クラスごとの精度
    print("\nクラスごとのテスト精度:")
    for i in range(num_classes):
        mask = y_test == i
        class_acc = np.mean(test_pred[mask] == y_test[mask])
        print(f"  クラス {i}: {class_acc*100:.1f}% ({np.sum(mask)} サンプル)")

    # いくつかの点で予測
    print("\n代表的な点での予測:")
    test_points = [
        ([0.5, 0.0], "クラス0領域"),
        ([0.0, 0.5], "クラス1領域"),
        ([-0.5, 0.0], "クラス2領域"),
        ([0.0, 0.0], "原点（境界）"),
    ]

    for point, description in test_points:
        x = Tensor([point])
        logits = model(x)
        probs = np.exp(logits.data) / np.sum(np.exp(logits.data))
        pred_class = np.argmax(probs)
        confidence = probs[0, pred_class]

        print(f"  {description:15s} ({point[0]:5.1f}, {point[1]:5.1f}) → "
              f"クラス {pred_class} (確信度: {confidence*100:.1f}%)")

    # ログを保存
    logger.save_csv("multiclass_classification_log.csv")
    logger.save_json("multiclass_classification_log.json")

    print("\n" + "=" * 60)
    print("訓練完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()

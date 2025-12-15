"""
CNNの例
シンプルな畳み込みニューラルネットワークで画像分類
"""

import numpy as np
from rim import Tensor, Logger
from rim.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, Dropout, BatchNorm2d
from rim.loss import CrossEntropyLoss
from rim.optim import Adam


class SimpleCNN(Module):
    """シンプルなCNNモデル"""

    def __init__(self, num_classes=10):
        super().__init__()
        # 畳み込み層1: 1 -> 32チャネル
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(32)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        # 畳み込み層2: 32 -> 64チャネル
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        # 全結合層
        # 入力: 28x28 -> pool1: 14x14 -> pool2: 7x7
        # 7 * 7 * 64 = 3136
        self.fc1 = Linear(7 * 7 * 64, 128)
        self.relu3 = ReLU()
        self.dropout = Dropout(p=0.5)
        self.fc2 = Linear(128, num_classes)

    def forward(self, x):
        # 畳み込み層1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 畳み込み層2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten
        batch_size = x.shape[0]
        x = Tensor(x.data.reshape(batch_size, -1), requires_grad=x.requires_grad)

        # 全結合層
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def generate_simple_image_data(n_samples=100, img_size=28, num_classes=10):
    """
    簡単な画像データを生成
    各クラスは異なるパターン（縦線、横線、対角線など）を持つ
    """
    np.random.seed(42)

    images = []
    labels = []

    samples_per_class = n_samples // num_classes

    for class_idx in range(num_classes):
        for _ in range(samples_per_class):
            img = np.zeros((img_size, img_size), dtype=np.float32)

            # クラスごとに異なるパターンを生成
            if class_idx == 0:
                # 縦線
                center = img_size // 2
                img[:, center-2:center+2] = 1.0
            elif class_idx == 1:
                # 横線
                center = img_size // 2
                img[center-2:center+2, :] = 1.0
            elif class_idx == 2:
                # 対角線（左上→右下）
                for i in range(img_size):
                    if 0 <= i < img_size:
                        img[i, i] = 1.0
            elif class_idx == 3:
                # 対角線（右上→左下）
                for i in range(img_size):
                    if 0 <= img_size-1-i < img_size:
                        img[i, img_size-1-i] = 1.0
            elif class_idx == 4:
                # 円
                center = img_size // 2
                radius = img_size // 4
                y, x = np.ogrid[:img_size, :img_size]
                mask = (x - center)**2 + (y - center)**2 <= radius**2
                img[mask] = 1.0
            elif class_idx == 5:
                # 四角形
                margin = img_size // 4
                img[margin:-margin, margin:-margin] = 1.0
            elif class_idx == 6:
                # 十字
                center = img_size // 2
                img[:, center-1:center+1] = 1.0
                img[center-1:center+1, :] = 1.0
            elif class_idx == 7:
                # 上半分
                img[:img_size//2, :] = 1.0
            elif class_idx == 8:
                # 下半分
                img[img_size//2:, :] = 1.0
            elif class_idx == 9:
                # チェッカーボード
                for i in range(0, img_size, 4):
                    for j in range(0, img_size, 4):
                        if (i // 4 + j // 4) % 2 == 0:
                            img[i:i+4, j:j+4] = 1.0

            # ノイズを追加
            noise = np.random.randn(img_size, img_size) * 0.1
            img = np.clip(img + noise, 0, 1)

            images.append(img)
            labels.append(class_idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # シャッフル
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # (batch, channel, height, width) の形状に変換
    images = images[:, np.newaxis, :, :]

    return images, labels


def main():
    print("=" * 60)
    print("CNN: シンプルな画像分類")
    print("=" * 60)

    # データ生成
    X_train, y_train = generate_simple_image_data(n_samples=500, num_classes=10)
    X_test, y_test = generate_simple_image_data(n_samples=100, num_classes=10)

    X_train_tensor = Tensor(X_train)
    y_train_tensor = Tensor(y_train.astype(np.float32))
    X_test_tensor = Tensor(X_test)

    print(f"訓練データ: {len(X_train)} サンプル")
    print(f"テストデータ: {len(X_test)} サンプル")
    print(f"画像サイズ: {X_train.shape[2]}x{X_train.shape[3]}")
    print(f"クラス数: 10")
    for i in range(10):
        count = np.sum(y_train == i)
        print(f"  クラス {i}: {count} サンプル")
    print()

    # モデル、損失関数、最適化器
    model = SimpleCNN(num_classes=10)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    print("モデル構造:")
    print("  Conv1:  1 -> 32 (3x3, padding=1) + BatchNorm + ReLU + MaxPool(2x2)")
    print("  Conv2: 32 -> 64 (3x3, padding=1) + BatchNorm + ReLU + MaxPool(2x2)")
    print("  FC1:   3136 -> 128 + ReLU + Dropout(0.5)")
    print("  FC2:   128 -> 10")
    print(f"  総パラメータ数: {len(list(model.parameters()))}\n")

    # ログ記録の初期化
    logger = Logger()

    # 訓練
    epochs = 50
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
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc*100:.1f}% | "
                  f"Test Acc: {test_acc*100:.1f}%")

    print("-" * 60)

    # 最終評価
    model.eval()
    test_logits = model(X_test_tensor)
    test_pred = np.argmax(test_logits.data, axis=1)
    test_acc = np.mean(test_pred == y_test)

    print(f"\n最終テスト精度: {test_acc*100:.2f}%")

    # クラスごとの精度
    print("\nクラスごとのテスト精度:")
    for i in range(10):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(test_pred[mask] == y_test[mask])
            print(f"  クラス {i}: {class_acc*100:.1f}% ({np.sum(mask)} サンプル)")

    # ログを保存
    logger.save_csv("cnn_log.csv")
    logger.save_json("cnn_log.json")

    print("\n" + "=" * 60)
    print("訓練完了！")
    print("=" * 60)


if __name__ == "__main__":
    main()

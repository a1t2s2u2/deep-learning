"""
訓練ログを記録するLoggerクラス
"""

import json
import csv
from typing import Dict, Any, List


class Logger:
    """訓練中のメトリクスを記録するクラス"""

    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def log(self, **metrics):
        """
        メトリクスを記録する

        Parameters
        ----------
        **metrics : dict
            記録するメトリクス（epoch=1, loss=0.5, acc=0.9など）
        """
        self.history.append(metrics)

    def save_csv(self, filepath: str):
        """
        CSVファイルとして保存

        Parameters
        ----------
        filepath : str
            保存先のファイルパス
        """
        if not self.history:
            print("警告: 記録されたログがありません")
            return

        # 全てのキーを収集
        keys = set()
        for entry in self.history:
            keys.update(entry.keys())
        keys = sorted(keys)

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

        print(f"ログをCSVに保存しました: {filepath}")

    def save_json(self, filepath: str):
        """
        JSONファイルとして保存

        Parameters
        ----------
        filepath : str
            保存先のファイルパス
        """
        if not self.history:
            print("警告: 記録されたログがありません")
            return

        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"ログをJSONに保存しました: {filepath}")

    def get_history(self) -> List[Dict[str, Any]]:
        """
        記録された履歴を取得

        Returns
        -------
        List[Dict[str, Any]]
            記録された全ての履歴
        """
        return self.history

    def get_metric(self, key: str) -> List[Any]:
        """
        特定のメトリクスの履歴を取得

        Parameters
        ----------
        key : str
            取得するメトリクスのキー

        Returns
        -------
        List[Any]
            指定されたメトリクスの値のリスト
        """
        return [entry.get(key) for entry in self.history if key in entry]

    def __len__(self) -> int:
        """記録されたエントリ数を返す"""
        return len(self.history)

    def __repr__(self) -> str:
        return f"Logger(entries={len(self.history)})"

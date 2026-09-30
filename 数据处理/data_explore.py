import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score
from typing import Optional, Dict, Tuple, Union
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AgriDataProcessor:
    """
    农业数据处理类，具有高级缺失值处理功能。

    功能特点：
    - 自动编码检测
    - 基于机器学习的缺失值填充
    - 填充质量评估
    - 全面的数据验证
    """

    def __init__(self, validation_threshold: float = 0.8):
        """
        初始化处理器，可配置参数。

        参数：
            validation_threshold: 填充质量可接受的最小分数 (0-1)
        """
        self.data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.data_path: Optional[Path] = None
        self.validation_threshold = validation_threshold
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: Union[str, Path], encoding: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        从CSV文件加载数据，支持自动编码检测。

        参数：
            file_path: CSV文件路径
            encoding: 可选的指定编码

        返回：
            加载的DataFrame或加载失败时返回None
        """
        try:
            self.data_path = Path(file_path).resolve()
            if not self.data_path.exists():
                self.logger.error(f"找不到文件: {self.data_path}")
                return None

            encodings = [encoding] if encoding else ['utf-8', 'gbk', 'gb2312', 'latin-1']

            for enc in encodings:
                try:
                    self.data = pd.read_csv(self.data_path, encoding=enc)
                    self.original_data = self.data.copy()
                    self.data.columns = self.data.columns.str.strip()

                    self.logger.info(f"成功加载文件: {self.data_path.name}")
                    self.logger.info(f"使用编码: {enc}")
                    self.logger.info(f"数据形状: {self.data.shape}")

                    # 验证数据结构
                    self._validate_data()
                    return self.data

                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"使用 {enc} 编码时出错: {str(e)}")
                    continue

            self.logger.error("尝试所有编码都未能成功读取文件")
            return None

        except Exception as e:
            self.logger.error(f"数据加载过程中发生意外错误: {str(e)}")
            return None

    def _validate_data(self) -> None:
        """验证加载的数据结构和质量。"""
        if self.data is not None:
            # 检查重复列
            duplicates = self.data.columns[self.data.columns.duplicated()]
            if not duplicates.empty:
                self.logger.warning(f"发现重复列: {list(duplicates)}")

            # 检查数据类型
            self.logger.info("列数据类型:")
            for col, dtype in self.data.dtypes.items():
                self.logger.info(f"{col}: {dtype}")

            # 检查缺失值
            missing = self.data.isnull().sum()
            if missing.any():
                self.logger.info("每列缺失值统计:")
                for col, count in missing[missing > 0].items():
                    self.logger.info(f"{col}: {count} ({(count / len(self.data) * 100):.2f}%)")

    def fill_missing_values_with_ml(self, n_neighbors: int = 5) -> Optional[pd.DataFrame]:
        """
        使用基于KNN的机器学习模型填充缺失值。

        参数：
            n_neighbors: KNN模型使用的邻居数量

        返回：
            填充后的DataFrame或处理失败时返回None
        """
        if self.data is None:
            self.logger.error("未加载数据，请先加载数据。")
            return None

        try:
            for column in self.data.columns:
                missing_count = self.data[column].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"\n处理列: {column} (缺失值数量: {missing_count})")

                    # 准备数据
                    non_missing_mask = ~self.data[column].isnull()

                    # 特征工程
                    features = self._prepare_features(column)

                    # 模型选择和训练
                    predictions = self._train_and_predict(
                        features,
                        column,
                        non_missing_mask,
                        n_neighbors
                    )

                    self.data.loc[~non_missing_mask, column] = predictions

            self.logger.info("缺失值填充完成")
            return self.data

        except Exception as e:
            self.logger.error(f"缺失值填充过程中发生错误: {str(e)}")
            return None

    def _prepare_features(self, target_column: str) -> pd.DataFrame:
        """准备用于机器学习模型的特征。"""
        features = self.data.drop(columns=[target_column])

        # 处理分类变量
        categorical_columns = features.select_dtypes(include=['object']).columns
        features = pd.get_dummies(features, columns=categorical_columns, drop_first=True)

        # 处理数值型缺失值
        numerical_imputer = SimpleImputer(strategy="mean")
        features = pd.DataFrame(
            numerical_imputer.fit_transform(features),
            columns=features.columns
        )

        return features

    def _train_and_predict(
            self,
            features: pd.DataFrame,
            target_column: str,
            non_missing_mask: pd.Series,
            n_neighbors: int
    ) -> np.ndarray:
        """训练模型并进行预测。"""
        is_numeric = self.data[target_column].dtype in [np.float64, np.int64]
        model = (KNeighborsRegressor(n_neighbors=n_neighbors) if is_numeric
                 else KNeighborsClassifier(n_neighbors=n_neighbors))

        known_features = features[non_missing_mask]
        unknown_features = features[~non_missing_mask]

        model.fit(known_features, self.data.loc[non_missing_mask, target_column])
        return model.predict(unknown_features)

    def evaluate_fill_quality(self) -> Dict[str, Dict[str, float]]:
        """
        评估值填充的质量。

        返回：
            包含每列评估指标的字典
        """
        if self.data is None or self.original_data is None:
            self.logger.error("没有可用于评估的数据")
            return {}

        evaluation_results = {}

        try:
            for column in self.data.columns:
                if self.original_data[column].isnull().sum() > 0:
                    results = self._evaluate_column(column)
                    evaluation_results[column] = results

                    # 记录评估结果
                    self.logger.info(f"\n列: {column}")
                    for metric, value in results.items():
                        self.logger.info(f"  {metric}: {value:.4f}")

                        # 对低质量填充发出警告
                        if value < self.validation_threshold:
                            self.logger.warning(
                                f"检测到{column}列的低质量填充 "
                                f"({metric}: {value:.4f})"
                            )

            return evaluation_results

        except Exception as e:
            self.logger.error(f"质量评估过程中发生错误: {str(e)}")
            return {}

    def _evaluate_column(self, column: str) -> Dict[str, float]:
        """评估单列的填充质量。"""
        non_missing_mask = ~self.original_data[column].isnull()
        true_values = self.original_data[column][non_missing_mask]
        filled_values = self.data[column][non_missing_mask]

        is_numeric = self.data[column].dtype in [np.float64, np.int64]

        if is_numeric:
            score = r2_score(true_values, filled_values)
            return {"R²分数": score}
        else:
            score = accuracy_score(true_values, filled_values)
            return {"准确率": score}

    def save_processed_data(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        将处理后的数据保存为CSV。

        参数：
            output_path: 可选的自定义输出文件路径
        """
        if self.data is None:
            self.logger.error("没有可保存的处理后数据")
            return

        try:
            if output_path is None:
                # 生成默认输出路径
                original_path = Path(self.data_path)
                output_path = original_path.parent / f"{original_path.stem}_processed{original_path.suffix}"

            output_path = Path(output_path)
            self.data.to_csv(output_path, index=False)
            self.logger.info(f"处理后的数据已保存至: {output_path}")

        except Exception as e:
            self.logger.error(f"保存处理后数据时发生错误: {str(e)}")
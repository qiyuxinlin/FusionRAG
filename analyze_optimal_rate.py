#!/usr/bin/env python3
"""
分析收集的 optimal rate 数据，拟合线性模型

使用方法:
    python analyze_optimal_rate.py

输入: optimal_rate_search/full_results.pkl
输出:
    - 线性回归模型参数
    - 特征重要性分析
    - 预测公式
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(data_path):
    """加载收集的数据"""
    with open(data_path, 'rb') as f:
        all_results = pickle.load(f)
    return all_results


def prepare_dataset(all_results):
    """准备训练数据集"""
    data = []

    for r in all_results:
        if r['min_correct_rate'] is None:
            continue  # 跳过所有 rate 都答错的

        features = r['attention_features']
        row = {
            'question_id': r['question_id'],
            'min_correct_rate': r['min_correct_rate'],
            **features
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def analyze_correlations(df, target='min_correct_rate'):
    """分析特征与目标的相关性"""
    print("\n" + "="*80)
    print("特征与 min_correct_rate 的相关性")
    print("="*80)

    # 排除非数值列
    exclude_cols = ['question_id', 'min_correct_rate', 'active_layers', 'attention_distribution']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    correlations = []
    for col in feature_cols:
        corr = df[col].corr(df[target])
        correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'特征':<30} {'相关系数':<12} {'方向'}")
    print("-" * 55)
    for feat, corr in correlations:
        direction = "↑ 需要更高 rate" if corr > 0 else "↓ 可以用更低 rate"
        print(f"{feat:<30} {corr:>8.4f}     {direction}")

    return correlations


def fit_linear_model(df, feature_cols=None):
    """拟合线性回归模型"""
    print("\n" + "="*80)
    print("线性回归模型拟合")
    print("="*80)

    if feature_cols is None:
        exclude = ['question_id', 'min_correct_rate', 'active_layers', 'attention_distribution', 'doc_len']
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df['min_correct_rate'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 尝试多种模型
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=0.1)': Ridge(alpha=0.1),
        'Lasso (α=0.01)': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    }

    print("\n模型交叉验证 (5-fold):")
    print("-" * 60)

    best_model = None
    best_score = -np.inf
    best_name = None

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
        mae_scores = -cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')

        print(f"{name:<25} R²={scores.mean():.4f}(±{scores.std():.4f})  MAE={mae_scores.mean():.4f}")

        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
            best_name = name

    print(f"\n最佳模型: {best_name}")

    # 用最佳模型拟合全部数据
    best_model.fit(X_scaled, y)

    # 输出系数
    print("\n" + "="*80)
    print(f"模型系数 ({best_name})")
    print("="*80)

    if hasattr(best_model, 'coef_'):
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': best_model.coef_,
            'abs_coef': np.abs(best_model.coef_)
        }).sort_values('abs_coef', ascending=False)

        print(f"\n{'特征':<30} {'系数':<12} {'影响'}")
        print("-" * 60)
        max_coef = coef_df['abs_coef'].max()
        for _, row in coef_df.iterrows():
            direction = "↑ rate" if row['coefficient'] > 0 else "↓ rate"
            if max_coef > 0:
                bar = '█' * int(abs(row['coefficient']) * 50 / max_coef)
            else:
                bar = ''
            print(f"{row['feature']:<30} {row['coefficient']:>8.4f}   {direction} {bar}")

        print(f"\n截距 (intercept): {best_model.intercept_:.4f}")

    return best_model, scaler, feature_cols


def generate_formula(model, scaler, feature_cols):
    """生成预测公式"""
    print("\n" + "="*80)
    print("预测公式")
    print("="*80)

    # 原始特征公式 (需要先标准化)
    print("\n方法 1: 使用标准化特征")
    print("-" * 40)
    print("# 先标准化特征")
    print("X_scaled = (X - mean) / std")
    print("# 然后预测")
    print(f"rate = {model.intercept_:.4f}", end="")
    for feat, coef in zip(feature_cols, model.coef_):
        if abs(coef) > 0.001:
            sign = " +" if coef > 0 else " "
            print(f"{sign}{coef:.4f} * {feat}_scaled", end="")
    print()

    # 直接使用原始特征的公式
    print("\n方法 2: 直接使用原始特征 (合并标准化)")
    print("-" * 40)

    # rate = intercept + sum(coef_i * (x_i - mean_i) / std_i)
    #      = intercept + sum(coef_i / std_i * x_i) - sum(coef_i * mean_i / std_i)
    #      = (intercept - sum(coef_i * mean_i / std_i)) + sum(coef_i / std_i * x_i)

    new_intercept = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)
    new_coefs = model.coef_ / scaler.scale_

    print(f"rate = {new_intercept:.6f}", end="")
    for feat, coef in zip(feature_cols, new_coefs):
        if abs(coef) > 0.0001:
            sign = " +" if coef > 0 else " "
            print(f"\n       {sign}{coef:.6f} * {feat}", end="")
    print()

    return {
        'feature_cols': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'coefficients': model.coef_.tolist(),
        'intercept': float(model.intercept_),
        'raw_coefficients': new_coefs.tolist(),
        'raw_intercept': float(new_intercept),
    }


def plot_analysis(df, model, scaler, feature_cols, output_dir):
    """绘制分析图表"""
    X = df[feature_cols].values
    y = df['min_correct_rate'].values
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 预测 vs 实际
    ax = axes[0, 0]
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect')
    ax.set_xlabel('Actual min_correct_rate')
    ax.set_ylabel('Predicted min_correct_rate')
    ax.set_title(f'Predicted vs Actual (R²={r2_score(y, y_pred):.3f})')
    ax.legend()

    # 2. 残差分布
    ax = axes[0, 1]
    residuals = y - y_pred
    ax.hist(residuals, bins=20, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution (MAE={np.abs(residuals).mean():.3f})')

    # 3. min_correct_rate 分布
    ax = axes[1, 0]
    ax.hist(y, bins=20, edgecolor='black')
    ax.set_xlabel('min_correct_rate')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of min_correct_rate')

    # 4. 特征重要性
    ax = axes[1, 1]
    importance = np.abs(model.coef_)
    sorted_idx = np.argsort(importance)[-10:]  # Top 10
    ax.barh(range(len(sorted_idx)), importance[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_xlabel('|Coefficient|')
    ax.set_title('Top 10 Feature Importance')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_plots.png'), dpi=150)
    print(f"\n图表已保存到: {output_dir}/analysis_plots.png")


def main():
    # 数据路径
    data_dir = "/mnt/data/wjh/FusionRAG/optimal_rate_search"
    data_path = os.path.join(data_dir, 'full_results.pkl')

    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        print("请先运行 collect_optimal_rate_data.py 收集数据")
        return

    print("="*80)
    print("加载数据")
    print("="*80)

    all_results = load_data(data_path)
    print(f"加载了 {len(all_results)} 个问题的数据")

    # 准备数据集
    df = prepare_dataset(all_results)
    print(f"有效样本数 (排除所有 rate 都答错的): {len(df)}")

    # 相关性分析
    correlations = analyze_correlations(df)

    # 选择特征 (排除一些冗余特征)
    feature_cols = [
        'peak_strength',
        'top10_concentration',
        'top20_concentration',
        'coverage_50_ratio',
        'coverage_70_ratio',
        'coverage_90_ratio',
        'normalized_entropy',
        'gini',
        'attention_std',
        'layer_consistency',
        'log_doc_len',
    ]

    # 确保特征存在
    feature_cols = [c for c in feature_cols if c in df.columns]

    # 拟合模型
    model, scaler, feature_cols = fit_linear_model(df, feature_cols)

    # 生成公式
    model_params = generate_formula(model, scaler, feature_cols)

    # 保存模型参数
    model_path = os.path.join(data_dir, 'linear_model.json')
    with open(model_path, 'w') as f:
        json.dump(model_params, f, indent=2)
    print(f"\n模型参数已保存到: {model_path}")

    # 绘制图表
    plot_analysis(df, model, scaler, feature_cols, data_dir)

    # 保存 DataFrame
    df.to_csv(os.path.join(data_dir, 'dataset.csv'), index=False)
    print(f"数据集已保存到: {data_dir}/dataset.csv")

    print("\n" + "="*80)
    print("完成!")
    print("="*80)


if __name__ == '__main__':
    main()

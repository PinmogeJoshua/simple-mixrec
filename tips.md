# 4. ALS推荐模型实现

根据您的数据特点和需求，我们将实现基于交替最小二乘法(ALS)的推荐系统模型。ALS特别适合处理稀疏的隐式反馈数据，如用户-餐厅签到记录。

## 4.1 ALS模型原理

```python
# ALS模型原理解析
"""
ALS（交替最小二乘法）原理：

1. 基本思想：将用户-物品交互矩阵分解为两个低维矩阵的乘积：R ≈ P × Q^T
   - R: 用户-物品评分/交互矩阵
   - P: 用户特征矩阵
   - Q: 物品特征矩阵

2. 对于隐式反馈，ALS使用加权损失函数：
   L = Σ c_ui (p_ui - x_ui)² + λ(||P||² + ||Q||²)
   其中：
   - c_ui: 置信度权重，通常设为 1 + α·r_ui
   - p_ui: 预测评分/喜好
   - x_ui: 观察到的隐式反馈值
   - λ: 正则化参数，防止过拟合

3. 优化方法：交替优化P和Q
   - 固定Q，优化P
   - 固定P，优化Q
   - 重复直到收敛

4. 优点：
   - 可并行计算
   - 适合稀疏数据
   - 处理隐式反馈效果好

5. 超参数：
   - rank: 隐因子维度
   - alpha: 置信度系数
   - regParam: 正则化参数
"""
```

## 4.2 环境准备与数据加载

```python
# 导入必要的库
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False

# 创建Spark会话
spark = SparkSession.builder \
    .appName("Restaurant Recommendation") \
    .getOrCreate()

# 加载数据
user_features_pd = pd.read_csv('data/user_features.csv')
interaction_features_pd = pd.read_csv('data/interaction_features.csv')

# 转换为Spark DataFrame
user_features = spark.createDataFrame(user_features_pd)
interaction_features = spark.createDataFrame(interaction_features_pd)

# 查看数据概要
print("交互数据概览:")
interaction_features.printSchema()
interaction_features.show(5)

print("用户数量:", interaction_features.select("user_id").distinct().count())
print("餐厅数量:", interaction_features.select("venue_id").distinct().count())
print("交互总数:", interaction_features.count())
```

## 4.3 数据预处理与转换

```python
# 为ALS模型准备数据
def prepare_als_data(interaction_df):
    """准备用于ALS模型的数据"""
    # 1. 确保评分/交互值存在
    if 'checkin_count' not in interaction_df.columns:
        raise ValueError("数据中需要有checkin_count列")
    
    # 2. 重命名列以符合ALS需求
    als_data = interaction_df.select(
        col('user_id').alias('user'),
        col('venue_id').alias('item'),
        col('checkin_count').alias('rating')
    )
    
    # 3. 处理缺失值，ALS不支持缺失值
    als_data = als_data.na.fill(1)  # 将缺失的交互次数设为1
    
    return als_data

# 准备训练数据
als_data = prepare_als_data(interaction_features)
print("ALS模型数据概览:")
als_data.show(5)

# 分割训练集和测试集
train_data, test_data = als_data.randomSplit([0.8, 0.2], seed=42)
print(f"训练集大小: {train_data.count()}")
print(f"测试集大小: {test_data.count()}")

# 可视化交互次数分布
plt.figure(figsize=(10, 6))
interaction_counts = interaction_features_pd['checkin_count'].value_counts().sort_index()
sns.barplot(x=interaction_counts.index, y=interaction_counts.values)
plt.title('用户-餐厅交互次数分布', fontsize=14)
plt.xlabel('交互次数', fontsize=12)
plt.ylabel('频次', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## 4.4 构建ALS基础模型

```python
# 构建ALS模型
def build_als_model(train_data, implicit=True, rank=10, reg_param=0.1, alpha=1.0, max_iter=10):
    """
    构建并训练ALS推荐模型
    
    参数:
        train_data: 训练数据
        implicit: 是否为隐式反馈数据
        rank: 隐因子维度
        reg_param: 正则化参数
        alpha: 置信度系数（仅用于隐式反馈）
        max_iter: 最大迭代次数
    
    返回:
        训练好的ALS模型
    """
    # 初始化ALS模型
    als = ALS(
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        rank=rank,
        regParam=reg_param,
        implicitPrefs=implicit,
        alpha=alpha,
        maxIter=max_iter,
        coldStartStrategy="drop"  # 处理冷启动问题
    )
    
    # 训练模型
    model = als.fit(train_data)
    
    return model

# 训练基础ALS模型
base_model = build_als_model(
    train_data,
    implicit=True,
    rank=10,
    reg_param=0.1,
    alpha=1.0,
    max_iter=10
)

print("基础ALS模型训练完成")
```

## 4.5 模型评估

```python
# 模型评估函数
def evaluate_als_model(model, test_data):
    """
    评估ALS模型性能
    
    参数:
        model: 训练好的ALS模型
        test_data: 测试数据
    
    返回:
        评估指标结果
    """
    # 对测试集进行预测
    predictions = model.transform(test_data)
    
    # 使用RMSE(均方根误差)评估
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    
    # 计算MAE(平均绝对误差)
    evaluator = RegressionEvaluator(
        metricName="mae",
        labelCol="rating",
        predictionCol="prediction"
    )
    mae = evaluator.evaluate(predictions)
    
    return {
        "RMSE": rmse,
        "MAE": mae
    }

# 评估基础模型
base_metrics = evaluate_als_model(base_model, test_data)
print("基础模型评估指标:")
for metric_name, value in base_metrics.items():
    print(f"{metric_name}: {value:.4f}")
```

## 4.6 生成推荐结果

```python
# 为所有用户生成推荐
def generate_recommendations(model, n_recommendations=10):
    """为所有用户生成推荐结果"""
    user_recs = model.recommendForAllUsers(n_recommendations)
    return user_recs

# 为特定用户生成推荐
def recommend_for_user(model, user_id, n_recommendations=10):
    """为特定用户生成推荐"""
    user_recs = model.recommendForUserSubset(
        spark.createDataFrame([(user_id,)], ["user"]),
        n_recommendations
    )
    return user_recs

# 生成推荐结果
all_recommendations = generate_recommendations(base_model, 10)
print("推荐结果示例:")
all_recommendations.show(5, truncate=False)

# 为示例用户生成推荐
sample_user_id = interaction_features_pd['user_id'].iloc[0]
user_recommendations = recommend_for_user(base_model, sample_user_id, 10)
print(f"用户 {sample_user_id} 的推荐结果:")
user_recommendations.show(truncate=False)
```

## 4.7 模型解释与可视化

```python
# 提取并可视化用户和物品特征矩阵
def visualize_embeddings(model, n_users=10, n_items=10):
    """可视化用户和物品的嵌入向量"""
    # 提取用户特征
    user_factors = model.userFactors.toPandas()
    user_factors = user_factors.sort_values('id').head(n_users)
    
    # 提取物品特征
    item_factors = model.itemFactors.toPandas()
    item_factors = item_factors.sort_values('id').head(n_items)
    
    # 将特征向量转换为二维空间进行可视化
    from sklearn.decomposition import PCA
    
    # 应用PCA
    pca = PCA(n_components=2)
    user_pca = pca.fit_transform(np.vstack(user_factors['features']))
    item_pca = pca.fit_transform(np.vstack(item_factors['features']))
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 用户嵌入向量
    plt.subplot(1, 2, 1)
    plt.scatter(user_pca[:, 0], user_pca[:, 1], c='blue', alpha=0.7)
    plt.title("用户嵌入向量的2D可视化", fontsize=14)
    plt.xlabel("主成分1")
    plt.ylabel("主成分2")
    
    # 物品嵌入向量
    plt.subplot(1, 2, 2)
    plt.scatter(item_pca[:, 0], item_pca[:, 1], c='red', alpha=0.7)
    plt.title("餐厅嵌入向量的2D可视化", fontsize=14)
    plt.xlabel("主成分1")
    plt.ylabel("主成分2")
    
    plt.tight_layout()
    plt.show()

# 可视化嵌入向量
visualize_embeddings(base_model)

# 分析用户-物品相似度
def analyze_user_item_similarity(model, user_id, top_n=5):
    """分析用户与推荐项目的相似度"""
    # 获取用户的嵌入向量
    user_vector = model.userFactors.filter(col('id') == user_id).select('features').collect()[0][0]
    
    # 获取所有物品的嵌入向量
    item_factors = model.itemFactors.toPandas()
    
    # 计算用户与所有物品的余弦相似度
    similarities = []
    for _, row in item_factors.iterrows():
        item_id = row['id']
        item_vector = row['features']
        
        # 计算余弦相似度
        similarity = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
        similarities.append((item_id, similarity))
    
    # 排序并获取最相似的物品
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_items = similarities[:top_n]
    
    return top_items

# 分析示例用户与推荐项目的相似度
sample_user_id = interaction_features_pd['user_id'].iloc[0]
top_similar_items = analyze_user_item_similarity(base_model, sample_user_id)
print(f"与用户 {sample_user_id} 最相似的餐厅:")
for item_id, similarity in top_similar_items:
    print(f"餐厅ID: {item_id}, 相似度: {similarity:.4f}")
```

## 4.8 保存模型

```python
# 保存模型以备后用
def save_als_model(model, path):
    """保存ALS模型"""
    model.save(path)
    print(f"模型已保存到 {path}")

# 加载保存的模型
def load_als_model(path):
    """加载保存的ALS模型"""
    from pyspark.ml.recommendation import ALSModel
    model = ALSModel.load(path)
    print(f"模型已从 {path} 加载")
    return model

# 创建模型保存目录
import os
os.makedirs('models', exist_ok=True)

# 保存基础模型
save_als_model(base_model, "models/base_als_model")
```

# 5. ALS模型超参数调优

为了获得最佳性能，我们需要对ALS模型的关键超参数进行系统的调优。这部分将实施不同的超参数调优策略，找到最优模型配置。

## 5.1 超参数调优的理论基础

```python
# 超参数调优理论基础
"""
ALS模型的主要超参数及其影响:

1. rank (隐因子维度):
   - 控制潜在因子空间的维度
   - 过小: 无法捕捉足够的数据模式
   - 过大: 容易过拟合，增加计算复杂度
   - 典型范围: 10-200

2. regParam (正则化参数):
   - 控制模型复杂度，防止过拟合
   - 过小: 可能导致过拟合
   - 过大: 可能无法捕捉足够的模式
   - 典型范围: 0.01-1.0

3. alpha (置信度系数):
   - 用于隐式反馈模型
   - 控制观察到的交互的置信度
   - 较大的值增加正样本的权重
   - 典型范围: 1.0-40.0

4. maxIter (最大迭代次数):
   - 控制算法的收敛过程
   - 需要足够大以确保收敛
   - 典型范围: 5-20次迭代

5. numBlocks (分区数):
   - 影响并行计算效率
   - 调整数据分区方式
   - 通常设置为集群核心数的倍数
"""
```

## 5.2 网格搜索超参数调优

```python
# 使用网格搜索进行超参数调优
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time

# 定义用于网格搜索的超参数网格
def create_param_grid(als):
    """创建超参数网格"""
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 20, 50]) \
        .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
        .addGrid(als.alpha, [1.0, 10.0, 40.0]) \
        .build()
    
    return param_grid

# 使用交叉验证进行网格搜索
def grid_search_cv(train_data, test_data):
    """使用交叉验证进行网格搜索超参数调优"""
    # 初始化ALS模型
    als = ALS(
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        implicitPrefs=True,
        coldStartStrategy="drop",
        maxIter=10  # 固定最大迭代次数以加速调优
    )
    
    # 创建超参数网格
    param_grid = create_param_grid(als)
    
    # 创建评估器
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    
    # 设置交叉验证
    cv = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,  # 3折交叉验证
        parallelism=4  # 并行度
    )
    
    print("开始网格搜索超参数调优...")
    start_time = time.time()
    
    # 运行交叉验证
    cv_model = cv.fit(train_data)
    
    end_time = time.time()
    print(f"超参数调优完成，耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    # 获取最佳模型
    best_model = cv_model.bestModel
    
    # 提取最佳参数
    best_rank = best_model._java_obj.parent().getRank()
    best_reg_param = best_model._java_obj.parent().getRegParam()
    best_alpha = best_model._java_obj.parent().getAlpha()
    
    print(f"最佳rank: {best_rank}")
    print(f"最佳regParam: {best_reg_param}")
    print(f"最佳alpha: {best_alpha}")
    
    # 在测试集上评估最佳模型
    test_metrics = evaluate_als_model(best_model, test_data)
    print("最佳模型在测试集上的表现:")
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    return best_model, {
        'rank': best_rank,
        'regParam': best_reg_param,
        'alpha': best_alpha
    }

# 执行网格搜索
best_grid_model, best_params = grid_search_cv(train_data, test_data)
```

## 5.3 随机搜索超参数调优

```python
# 随机搜索超参数调优
import random

def random_search(train_data, test_data, n_iter=15):
    """
    使用随机搜索进行超参数调优
    
    参数:
        train_data: 训练数据
        test_data: 测试数据
        n_iter: 随机搜索迭代次数
    
    返回:
        最佳模型和最佳参数
    """
    # 定义超参数搜索空间
    param_space = {
        'rank': [10, 15, 20, 30, 50, 75, 100],
        'regParam': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        'alpha': [0.1, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0]
    }
    
    best_rmse = float('inf')
    best_model = None
    best_params = {}
    results = []
    
    print("开始随机搜索超参数调优...")
    start_time = time.time()
    
    for i in range(n_iter):
        # 随机选择参数
        rank = random.choice(param_space['rank'])
        reg_param = random.choice(param_space['regParam'])
        alpha = random.choice(param_space['alpha'])
        
        print(f"\n迭代 {i+1}/{n_iter}:")
        print(f"尝试参数: rank={rank}, regParam={reg_param}, alpha={alpha}")
        
        # 训练模型
        model = build_als_model(
            train_data, 
            implicit=True,
            rank=rank, 
            reg_param=reg_param, 
            alpha=alpha,
            max_iter=10
        )
        
        # 评估模型
        metrics = evaluate_als_model(model, test_data)
        rmse = metrics['RMSE']
        
        print(f"RMSE: {rmse:.4f}")
        
        # 记录结果
        results.append({
            'rank': rank,
            'regParam': reg_param,
            'alpha': alpha,
            'RMSE': rmse
        })
        
        # 更新最佳模型
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = {
                'rank': rank,
                'regParam': reg_param,
                'alpha': alpha
            }
            print(f"发现新的最佳模型! RMSE: {best_rmse:.4f}")
    
    end_time = time.time()
    print(f"\n随机搜索完成，耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    # 将结果转换为DataFrame并排序
    results_df = pd.DataFrame(results).sort_values('RMSE')
    print("\n随机搜索结果 (按RMSE排序):")
    print(results_df.head(10))
    
    print("\n最佳参数:")
    print(f"rank: {best_params['rank']}")
    print(f"regParam: {best_params['regParam']}")
    print(f"alpha: {best_params['alpha']}")
    print(f"RMSE: {best_rmse:.4f}")
    
    return best_model, best_params, results_df

# 执行随机搜索
best_random_model, best_random_params, random_search_results = random_search(train_data, test_data)
```

## 5.4 贝叶斯优化超参数调优

```python
# 贝叶斯优化超参数调优
# 注意：需要安装 hyperopt 库 (pip install hyperopt)
try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    import numpy as np
    
    def objective(params):
        """贝叶斯优化的目标函数"""
        rank = int(params['rank'])
        reg_param = params['reg_param']
        alpha = params['alpha']
        
        print(f"\n尝试参数: rank={rank}, regParam={reg_param}, alpha={alpha}")
        
        # 训练模型
        model = build_als_model(
            train_data, 
            implicit=True,
            rank=rank, 
            reg_param=reg_param, 
            alpha=alpha,
            max_iter=10
        )
        
        # 评估模型
        metrics = evaluate_als_model(model, test_data)
        rmse = metrics['RMSE']
        
        print(f"RMSE: {rmse:.4f}")
        
        return {
            'loss': rmse,
            'status': STATUS_OK,
            'model': model,
            'params': {
                'rank': rank,
                'regParam': reg_param,
                'alpha': alpha
            }
        }
    
    def bayesian_optimization(n_iter=20):
        """使用贝叶斯优化进行超参数调优"""
        # 定义参数空间
        space = {
            'rank': hp.quniform('rank', 5, 100, 5),  # 从5到100，步长为5
            'reg_param': hp.loguniform('reg_param', np.log(0.001), np.log(1.0)),  # 对数均匀分布
            'alpha': hp.loguniform('alpha', np.log(0.1), np.log(100.0))  # 对数均匀分布
        }
        
        # 存储所有试验
        trials = Trials()
        
        print("开始贝叶斯优化超参数调优...")
        start_time = time.time()
        
        # 使用TPE算法进行贝叶斯优化
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=n_iter
        )
        
        end_time = time.time()
        print(f"\n贝叶斯优化完成，耗时: {(end_time - start_time)/60:.2f} 分钟")
        
        # 获取最佳参数和模型
        best_trial = min(trials.results, key=lambda x: x['loss'])
        best_model = best_trial['model']
        best_params = best_trial['params']
        best_rmse = best_trial['loss']
        
        print("\n贝叶斯优化最佳参数:")
        print(f"rank: {best_params['rank']}")
        print(f"regParam: {best_params['regParam']}")
        print(f"alpha: {best_params['alpha']}")
        print(f"RMSE: {best_rmse:.4f}")
        
        # 准备结果DataFrame
        results = []
        for i, trial in enumerate(trials.results):
            results.append({
                'iteration': i + 1,
                'rank': trial['params']['rank'],
                'regParam': trial['params']['regParam'],
                'alpha': trial['params']['alpha'],
                'RMSE': trial['loss']
            })
        
        results_df = pd.DataFrame(results).sort_values('RMSE')
        print("\n贝叶斯优化结果 (按RMSE排序):")
        print(results_df.head(10))
        
        return best_model, best_params, results_df
    
    # 执行贝叶斯优化
    best_bayes_model, best_bayes_params, bayes_results = bayesian_optimization()

except ImportError:
    print("未安装hyperopt库，跳过贝叶斯优化部分")
    print("可以使用 'pip install hyperopt' 安装该库")
```

## 5.5 调优结果比较与可视化

```python
# 比较不同调优方法的结果
def compare_tuning_methods(grid_params, random_params, bayes_params=None):
    """比较不同调优方法的结果"""
    # 创建比较表
    methods = ['网格搜索', '随机搜索']
    params = [grid_params, random_params]
    
    if bayes_params:
        methods.append('贝叶斯优化')
        params.append(bayes_params)
    
    # 使用最佳参数训练模型并评估
    results = []
    for method, param in zip(methods, params):
        # 训练模型
        model = build_als_model(
            train_data, 
            implicit=True,
            rank=param['rank'], 
            reg_param=param['regParam'], 
            alpha=param['alpha'],
            max_iter=15  # 使用更多迭代次数进行最终训练
        )
        
        # 评估模型
        metrics = evaluate_als_model(model, test_data)
        
        # 记录结果
        results.append({
            '调优方法': method,
            'rank': param['rank'],
            'regParam': param['regParam'],
            'alpha': param['alpha'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE']
        })
    
    # 创建比较DataFrame
    comparison_df = pd.DataFrame(results)
    
    # 可视化比较结果
    plt.figure(figsize=(12, 8))
    
    # RMSE比较
    plt.subplot(2, 1, 1)
    sns.barplot(x='调优方法', y='RMSE', data=comparison_df)
    plt.title('不同调优方法的RMSE比较', fontsize=14)
    plt.ylabel('RMSE (越低越好)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE比较
    plt.subplot(2, 1, 2)
    sns.barplot(x='调优方法', y='MAE', data=comparison_df)
    plt.title('不同调优方法的MAE比较', fontsize=14)
    plt.ylabel('MAE (越低越好)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    return comparison_df

# 比较不同调优方法
try:
    comparison_results = compare_tuning_methods(
        best_params, 
        best_random_params, 
        best_bayes_params if 'best_bayes_params' in locals() else None
    )
    print("\n调优方法比较结果:")
    print(comparison_results)
except:
    print("无法比较所有调优方法，可能是因为某些方法未执行")
```

## 5.6 超参数敏感性分析

```python
# 超参数敏感性分析
def parameter_sensitivity_analysis():
    """分析模型对不同超参数的敏感性"""
    # 1. rank参数敏感性分析
    ranks = [5, 10, 20, 50, 100]
    rank_results = []
    
    # 固定其他参数
    reg_param = best_random_params['regParam']
    alpha = best_random_params['alpha']
    
    print("\n分析rank参数敏感性...")
    for rank in ranks:
        model = build_als_model(
            train_data, 
            implicit=True,
            rank=rank, 
            reg_param=reg_param, 
            alpha=alpha,
            max_iter=10
        )
        metrics = evaluate_als_model(model, test_data)
        rank_results.append({
            'rank': rank,
            'RMSE': metrics['RMSE']
        })
        print(f"rank={rank}, RMSE={metrics['RMSE']:.4f}")
    
    # 2. regParam参数敏感性分析
    reg_params = [0.001, 0.01, 0.1, 0.5, 1.0]
    reg_param_results = []
    
    # 固定其他参数
    rank = best_random_params['rank']
    
    print("\n分析regParam参数敏感性...")
    for reg_param in reg_params:
        model = build_als_model(
            train_data, 
            implicit=True,
            rank=rank, 
            reg_param=reg_param, 
            alpha=alpha,
            max_iter=10
        )
        metrics = evaluate_als_model(model, test_data)
        reg_param_results.append({
            'regParam': reg_param,
            'RMSE': metrics['RMSE']
        })
        print(f"regParam={reg_param}, RMSE={metrics['RMSE']:.4f}")
    
    # 3. alpha参数敏感性分析
    alphas = [0.1, 1.0, 5.0, 10.0, 40.0]
    alpha_results = []
    
    # 固定其他参数
    reg_param = best_random_params['regParam']
    
    print("\n分析alpha参数敏感性...")
    for alpha in alphas:
        model = build_als_model(
            train_data, 
            implicit=True,
            rank=rank, 
            reg_param=reg_param, 
            alpha=alpha,
            max_iter=10
        )
        metrics = evaluate_als_model(model, test_data)
        alpha_results.append({
            'alpha': alpha,
            'RMSE': metrics['RMSE']
        })
        print(f"alpha={alpha}, RMSE={metrics['RMSE']:.4f}")
    
    # 可视化参数敏感性
    plt.figure(figsize=(15, 5))
    
    # rank敏感性
    plt.subplot(1, 3, 1)
    sns.lineplot(x='rank', y='RMSE', data=pd.DataFrame(rank_results), marker='o')
    plt.title('rank参数敏感性分析', fontsize=14)
    plt.xlabel('rank值')
    plt.ylabel('RMSE (越低越好)')
    plt.grid(linestyle='--', alpha=0.7)
    
    # regParam敏感性
    plt.subplot(1, 3, 2)
    sns.lineplot(x='regParam', y='RMSE', data=pd.DataFrame(reg_param_results), marker='o')
    plt.title('regParam参数敏感性分析', fontsize=14)
    plt.xlabel('regParam值')
    plt.ylabel('RMSE (越低越好)')
    plt.grid(linestyle='--', alpha=0.7)
    plt.xscale('log')  # 对数刻度
    
    # alpha敏感性
    plt.subplot(1, 3, 3)
    sns.lineplot(x='alpha', y='RMSE', data=pd.DataFrame(alpha_results), marker='o')
    plt.title('alpha参数敏感性分析', fontsize=14)
    plt.xlabel('alpha值')
    plt.ylabel('RMSE (越低越好)')
    plt.grid(linestyle='--', alpha=0.7)
    plt.xscale('log')  # 对数刻度
    
    plt.tight_layout()
    plt.show()
    
    return {
        'rank_sensitivity': pd.DataFrame(rank_results),
        'regParam_sensitivity': pd.DataFrame(reg_param_results),
        'alpha_sensitivity': pd.DataFrame(alpha_results)
    }

# 执行超参数敏感性分析
sensitivity_results = parameter_sensitivity_analysis()
```

## 5.7 最终模型训练与保存

```python
# 使用最佳参数训练最终模型
def train_final_model(best_params):
    """使用最佳参数训练最终模型"""
    print("\n使用最佳参数训练最终模型...")
    final_model = build_als_model(
        train_data = spark.createDataFrame(interaction_features_pd).select(
            col('user_id').alias('user'),
            col('venue_id').alias('item'),
            col('checkin_count').alias('rating')
        ).na.fill(1),  # 使用所有数据
        implicit=True,
        rank=best_params['rank'],
        reg_param=best_params['regParam'],
        alpha=best_params['alpha'],
        max_iter=20  # 增加迭代次数以确保收敛
    )
    
    print("最终模型训练完成!")
    return final_model

# 训练最终模型
final_model = train_final_model(best_random_params)  # 使用随机搜索的最佳参数

# 保存最终模型
save_als_model(final_model, "models/final_als_model")

# 生成一些最终模型的推荐示例
print("\n使用最终模型生成推荐样例:")
all_users = interaction_features_pd['user_id'].unique()
sample_users = np.random.choice(all_users, min(5, len(all_users)), replace=False)

for user_id in sample_users:
    recommendations = recommend_for_user(final_model, user_id, 10)
    user_recs = recommendations.toPandas()
    
    if not user_recs.empty:
        recs = user_recs.iloc[0]['recommendations']
        print(f"\n用户 {user_id} 的推荐餐厅:")
        for i, rec in enumerate(recs, 1):
            print(f"{i}. 餐厅ID: {rec.item}, 预测得分: {rec.rating:.4f}")

# 关闭Spark会话
spark.stop()
print("\nALS模型超参数调优完成，并保存了最终模型。")
```

这套完整的代码实现了ALS推荐模型的构建和系统化超参数调优，包括网格搜索、随机搜索和贝叶斯优化三种调优策略，并进行了详细的敏感性分析和结果可视化。通过这些步骤，我们能够找到最适合餐厅推荐数据的模型配置，并提供高质量的个性化推荐结果。
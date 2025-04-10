我理解您希望了解这些函数应该如何添加到您的代码中。基于您的`rec.ipynb`文件和当前上下文，我将详细说明如何实现这些函数，以及它们应该放在哪个位置。

## 首先，添加`get_similar_users_recommendations`函数

这个函数应该添加在"4.3 基于用户特征的推荐"部分，在`recommend_by_similar_users`函数之后：

```python
def get_similar_users_recommendations(user_id, user_features_df, interaction_df, N=10):
    """
    基于用户相似度的协作过滤推荐
    
    参数:
    - user_id: 目标用户ID
    - user_features_df: 用户特征DataFrame
    - interaction_df: 交互数据DataFrame
    - N: 推荐数量
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 检查用户是否存在
    if user_id not in user_features_df['user_id'].values:
        print(f"用户ID {user_id} 不在数据集中")
        return []
    
    # 准备用户特征
    user_features = user_features_df.set_index('user_id')
    features_columns = ['total_checkins', 'venue_count', 'loyalty_score', 
                        'tip_count', 'avg_tip_length', 'avg_sentiment']
    
    # 有些列可能不存在，选择存在的列
    features_columns = [col for col in features_columns if col in user_features.columns]
    
    # 如果没有足够的特征，直接返回
    if not features_columns:
        print("没有足够的用户特征进行比较")
        return []
    
    # 过滤掉目标用户ID
    other_users = user_features.drop(user_id, errors='ignore')
    
    # 获取目标用户和其他用户的特征向量
    target_user_vector = user_features.loc[[user_id]][features_columns].values
    other_users_vectors = other_users[features_columns].values
    
    # 计算余弦相似度
    similarities = cosine_similarity(target_user_vector, other_users_vectors)[0]
    
    # 将相似度与用户ID对应
    user_similarities = list(zip(other_users.index, similarities))
    
    # 按相似度排序
    user_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 显示最相似的前5名用户
    print("\n相似用户:")
    for i, (similar_user, similarity) in enumerate(user_similarities[:5]):
        user_checkins = interaction_df[interaction_df['user_id'] == similar_user]['checkin_count'].sum()
        user_venues = interaction_df[interaction_df['user_id'] == similar_user]['venue_id'].nunique()
        print(f"  {i+1}. 用户ID: {similar_user}, 相似度: {similarity:.4f}, 签到: {user_checkins}, 餐厅数: {user_venues}")
    
    # 获取目标用户已访问的餐厅
    user_visited_venues = set(interaction_df[interaction_df['user_id'] == user_id]['venue_id'])
    
    print(f"用户已访问 {len(user_visited_venues)} 个餐厅")
    
    # 从相似用户中获取推荐
    venue_scores = {}
    for similar_user, similarity in user_similarities[:20]:  # 取前20个相似用户
        # 获取相似用户访问过的餐厅
        similar_user_venues = interaction_df[interaction_df['user_id'] == similar_user]
        
        for _, row in similar_user_venues.iterrows():
            venue_id = row['venue_id']
            
            # 过滤掉目标用户已访问过的餐厅
            if venue_id in user_visited_venues:
                continue
            
            # 使用签到次数和情感评分加权
            weight = row['checkin_count'] * (row['avg_sentiment'] + 1) if not np.isnan(row['avg_sentiment']) else row['checkin_count']
            score = similarity * weight
            
            # 累加分数
            if venue_id in venue_scores:
                venue_scores[venue_id] += score
            else:
                venue_scores[venue_id] = score
    
    # 转换为列表并排序
    recommendations = [(venue_id, score) for venue_id, score in venue_scores.items()]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"找到 {len(recommendations)} 条推荐")
    
    return recommendations[:N]
```

## 然后，添加`get_content_based_recommendations`函数

这个函数应该添加在"4.4 基于用户和餐厅特征的内容推荐"部分，替换或添加到`get_enhanced_content_recommendations`函数之后：

```python
def get_content_based_recommendations(user_id, user_features_df, venue_features_df, interaction_df, N=10):
    """
    基于用户和餐厅特征的内容推荐
    
    参数:
    - user_id: 用户ID
    - user_features_df: 用户特征DataFrame
    - venue_features_df: 餐厅特征DataFrame
    - interaction_df: 交互特征DataFrame
    - N: 推荐数量
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 检查用户是否存在
    if user_id not in user_features_df['user_id'].values:
        print(f"用户ID {user_id} 不在数据集中")
        return []
    
    # 获取用户访问过的餐厅
    user_venues = interaction_df[interaction_df['user_id'] == user_id]['venue_id'].unique()
    
    # 检查用户是否有访问记录
    if len(user_venues) == 0:
        print(f"用户 {user_id} 没有访问记录，无法基于内容推荐")
        return []
    
    print(f"用户访问过 {len(user_venues)} 个餐厅，其中 {len(set(user_venues) & set(venue_features_df['venue_id']))} 个有特征信息")
    
    # 获取用户访问过的餐厅特征
    visited_venues_features = venue_features_df[venue_features_df['venue_id'].isin(user_venues)]
    
    # 如果没有足够的特征信息，返回空列表
    if len(visited_venues_features) == 0:
        print("没有足够的特征信息进行内容推荐")
        return []
    
    # 计算用户偏好的特征向量（取平均）
    tag_columns = [col for col in venue_features_df.columns if col.startswith('tag_')]
    numeric_columns = ['total_checkins', 'unique_visitors', 'attraction_score', 'tip_count', 'avg_sentiment']
    
    # 标准化数值特征
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    venue_features_df_normalized = venue_features_df.copy()
    venue_features_df_normalized[numeric_columns] = scaler.fit_transform(venue_features_df[numeric_columns])
    
    # 计算用户历史餐厅的平均特征
    user_preference_vector = visited_venues_features[tag_columns + numeric_columns].mean().values.reshape(1, -1)
    
    # 计算所有餐厅与用户偏好的相似度
    all_venue_features = venue_features_df_normalized[tag_columns + numeric_columns].values
    similarities = cosine_similarity(user_preference_vector, all_venue_features)[0]
    
    # 创建餐厅ID和相似度的映射
    venue_similarities = list(zip(venue_features_df['venue_id'], similarities))
    
    # 过滤掉用户已访问过的餐厅
    venue_similarities = [(v_id, sim) for v_id, sim in venue_similarities if v_id not in user_venues]
    
    # 按相似度排序并选择前N个
    venue_similarities.sort(key=lambda x: x[1], reverse=True)
    
    return venue_similarities[:N]
```

## 接着，添加`diversify_recommendations`函数

这个函数应该添加在"5.1 混合推荐模型构建"部分，在`mixrec_hybrid_recommendations_enhanced`函数之后：

```python
def diversify_recommendations(recommendations, venue_features_df, N=10):
    """
    增加推荐列表的多样性
    
    参数:
    - recommendations: 初始推荐列表，包含(venue_id, score, source, original_scores)
    - venue_features_df: 餐厅特征DataFrame
    - N: 最终推荐数量
    """
    try:
        # 确保有足够的推荐
        if len(recommendations) <= N:
            return recommendations
        
        # 获取餐厅标签
        tag_columns = [col for col in venue_features_df.columns if col.startswith('tag_')]
        venue_tags = {}
        
        for venue_id in [r[0] for r in recommendations]:
            venue_data = venue_features_df[venue_features_df['venue_id'] == venue_id]
            if not venue_data.empty:
                tags = []
                for tag in tag_columns:
                    if venue_data[tag].values[0] == 1:
                        tags.append(tag[4:])  # 去掉'tag_'前缀
                venue_tags[venue_id] = set(tags)
            else:
                venue_tags[venue_id] = set()
        
        # 多样性推荐算法
        final_diverse = []
        
        # 首先添加得分最高的推荐
        final_diverse.append(recommendations[0])
        selected_tags = venue_tags.get(recommendations[0][0], set())
        
        remaining = recommendations[1:]
        
        # 继续选择最多样的推荐
        while len(final_diverse) < N and remaining:
            # 为每个候选计算多样性分数
            diversity_scores = []
            
            for i, (venue_id, score, source, original_scores) in enumerate(remaining):
                # 计算与已选择项的标签交集
                venue_tag_set = venue_tags.get(venue_id, set())
                
                # 如果没有标签，给予较低的多样性分数
                if not venue_tag_set:
                    overlap_ratio = 0.5  # 中等多样性
                else:
                    # 计算标签重叠比例
                    overlap = len(selected_tags & venue_tag_set)
                    total = len(selected_tags | venue_tag_set) if len(selected_tags | venue_tag_set) > 0 else 1
                    overlap_ratio = 1 - (overlap / total)  # 多样性分数 (1 = 完全不同，0 = 完全相同)
                
                diversity_scores.append((i, overlap_ratio))
            
            # 选择多样性最高的选项
            diversity_scores.sort(key=lambda x: x[1], reverse=True)
            best_idx = diversity_scores[0][0]
            
            # 添加差异最大的推荐
            best_rec = remaining.pop(best_idx)
            final_diverse.append(best_rec)
            # 更新已选标签集合
            if best_rec[0] in venue_tags:
                selected_tags.update(venue_tags[best_rec[0]])
        
        # 如果多样化后的推荐不足，从原列表补充
        while len(final_diverse) < N and remaining:
            final_diverse.append(remaining.pop(0))
            
        return final_diverse
    except Exception as e:
        print(f"多样性处理异常: {e}")
        # 出错时返回原始推荐
        return recommendations[:N]
```

## 最后，完善`get_improved_als_recommendations`函数

如果您的代码中已经有这个函数，只是需要补充完整，请在"4.1 协同过滤模型"部分找到这个函数并确保其实现与我前面提供的一致。如果没有，您应该添加在这个部分：

```python
def get_improved_als_recommendations(model, user_id, user_map, venue_map, venue_index_to_id, 
                                    interaction_matrix, N=10, diversity_factor=0.2):
    """
    获取改进的ALS推荐
    
    参数:
    - diversity_factor: 多样性调节因子，0-1之间，值越大推荐多样性越高
    """
    if user_id not in user_map:
        print(f"用户ID {user_id} 不在训练数据中")
        return []
    
    user_idx = user_map[user_id]
    
    # 获取用户向量
    user_vector = model.user_factors[user_idx]
    
    # 计算所有物品的预测分数
    item_scores = model.item_factors.dot(user_vector)
    
    # 获取用户已交互的物品索引
    interacted_items = interaction_matrix[user_idx].indices
    
    # 过滤已交互的物品，将其分数设为负无穷
    item_scores[interacted_items] = -np.inf
    
    # 选择前N*2个物品，用于增加多样性选择空间
    top_items_indices = np.argsort(-item_scores)[:int(N*2)]
    candidate_items = [(venue_index_to_id[idx], item_scores[idx]) for idx in top_items_indices]
    
    # 应用多样性增强算法
    if diversity_factor > 0:
        selected_items = []
        remaining_candidates = candidate_items.copy()
        
        # 先选择得分最高的物品
        selected_items.append(remaining_candidates.pop(0))
        
        # 然后交替选择高分和多样性物品
        while len(selected_items) < N and remaining_candidates:
            # 计算剩余候选项与已选项的相似度
            candidate_scores = []
            for cand_id, cand_score in remaining_candidates:
                # 多样性得分 - 与已选项的最大相似度的负值
                try:
                    diversity_score = -max([model.item_factors[venue_map[v_id]].dot(model.item_factors[venue_map[cand_id]])
                                           for v_id, _ in selected_items]) if selected_items else 0
                except KeyError:
                    # 处理venue_map中可能缺失的ID
                    diversity_score = 0
                
                # 组合得分 = 原始得分 * (1-diversity_factor) + 多样性得分 * diversity_factor
                combined_score = cand_score * (1-diversity_factor) + diversity_score * diversity_factor
                candidate_scores.append((cand_id, cand_score, combined_score))
            
            # 选择组合得分最高的项
            candidate_scores.sort(key=lambda x: x[2], reverse=True)
            best_candidate = candidate_scores[0]
            selected_items.append((best_candidate[0], best_candidate[1]))
            
            # 从候选列表中移除已选项
            remaining_candidates = [item for item in remaining_candidates if item[0] != best_candidate[0]]
        
        return selected_items[:N]
    else:
        # 不应用多样性时，直接返回得分最高的N个物品
        return candidate_items[:N]
```

这样，您就完成了所有必要函数的添加，可以支持`mixrec_hybrid_recommendations_enhanced`函数的正常运行。请确保这些函数都添加在适当的位置，并且导入了所有必要的库。
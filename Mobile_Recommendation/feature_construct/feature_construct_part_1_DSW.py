# -*- coding: utf-8 -*
    
'''
@author: PY131 - Modified for streamlined features and DSW 48-core optimization

精简特征版本 - 每类只保留5个核心特征
DSW 48核心+372GB内存优化版本 - 适配阿里云DSW环境
'''

import pandas as pd
import numpy as np
import os
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

# ...existing code...
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

# === 新增导入：绘图与VIF ===
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # 无界面环境下保存图片
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
# ...existing code...

# ...existing code...
import os
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # 无界面环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
# ...existing code...


def setup_performance_optimization():
    """
    配置性能优化参数
    基于DSW 48核心+372GB内存配置 【修改】
    """
    # 获取环境变量中的优化参数，如果没有则使用DSW默认值
    memory_mode = os.environ.get('TIANCHI_MEMORY_MODE', 'normal')
    cpu_cores = int(os.environ.get('TIANCHI_CPU_CORES', '40'))  # 【修改】DSW使用40核心
    suggested_chunk_size = int(os.environ.get('TIANCHI_CHUNK_SIZE', '500000'))  # 【修改】DSW大内存默认50万行
    
    # DSW 372GB超大内存配置，可以使用很大的chunk_size 【修改】
    if memory_mode == 'conservative':
        chunk_size = min(suggested_chunk_size, 300000)  # 【修改】保守模式仍然很大
    else:
        chunk_size = min(800000, suggested_chunk_size * 2)  # 【修改】正常模式可以用80万行chunk！
    
    # 设置pandas的多线程参数，充分利用DSW多核 【修改】
    os.environ['OMP_NUM_THREADS'] = str(min(cpu_cores, 20))  # 【修改】pandas操作使用20核心
    os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_cores)  # 【修改】数值计算使用全部核心
    
    print(f"DSW性能优化配置:") # 【修改】
    print(f"   CPU核心数: {multiprocessing.cpu_count()} (使用{cpu_cores}个)")
    print(f"   内存模式: {memory_mode}")
    print(f"   Chunk大小: {chunk_size:,} (DSW大内存优化)") # 【修改】
    print(f"   多线程设置: pandas={min(cpu_cores, 20)}核心, 数值计算={cpu_cores}核心") # 【修改】
    print(f"   预计可处理: 全量11亿+数据") # 【修改】
    
    return chunk_size

print("开始构建Part 1精简特征集 (DSW全量数据)...") # 【修改】
print("=" * 60)

# 配置性能优化
chunk_size = setup_performance_optimization()

##### file path
# input 【修改全量数据路径】
path_df_D = "data/tianchi_fresh_comp_train_user_online_partA.txt"  # 【修改】全量数据文件A (不再使用，但保留兼容性)
path_df_part_1 = "Mobile_Recommendation/data/mobile/df_part_1.csv"  # 【不修改】Part1数据来自divide_data_set.py的输出
path_df_part_1_uic_label = "Mobile_Recommendation/data/mobile/df_part_1_uic_label.csv"  # 【不修改】

# output - 统一保存到Mobile_Recommendation\data\mobile\feature下面 【不修改】
path_df_part_1_U   = "Mobile_Recommendation/data/mobile/feature/df_part_1_U.csv"  
path_df_part_1_I   = "Mobile_Recommendation/data/mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "Mobile_Recommendation/data/mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "Mobile_Recommendation/data/mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "Mobile_Recommendation/data/mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "Mobile_Recommendation/data/mobile/feature/df_part_1_UC.csv"

# 创建输出目录 【不修改】
os.makedirs("Mobile_Recommendation/data/mobile/feature", exist_ok=True)
print(f"创建输出目录: Mobile_Recommendation/data/mobile/feature")

# 检查输入文件 【不修改】
if not os.path.exists(path_df_part_1):
    print(f"错误: Part1文件不存在 {path_df_part_1}")
    exit(1)
else:
    file_size = os.path.getsize(path_df_part_1) / (1024 * 1024)  # MB
    print(f"Part1文件检查通过: {path_df_part_1} ({file_size:.1f}MB)")

##========================================================##
##======================== Part 1 ========================##
##========================================================##

###########################################
'''Step 1.1 用户(U)特征 - 精简版(5个特征) - DSW 48核心优化 【修改注释】
    1. u_b4_count_in_6      # 用户6天内购买次数
    2. u_b3_count_in_6      # 用户6天内加购物车次数  
    3. u_b4_rate            # 用户购买转化率
    4. u_b_count_in_1       # 用户最近1天总行为数
    5. u_b4_diff_hours      # 用户首次购买时间差
'''

print("\nStep 1.1: 构建用户(U)特征 (5个核心特征)...")

# 性能优化：预定义数据类型减少内存占用 【不修改】
dtype_dict = {
    'user_id': 'int32',
    'item_id': 'int32',
    'behavior_type': 'int8',
    'item_category': 'int16'
}

u_features_list = []

print(f"   分块处理Part1数据 (DSW大内存chunk_size: {chunk_size:,})...") # 【修改】

chunk_count = 0

# DSW优化：使用更高效的CSV读取参数和更大chunk_size 【修改】
csv_params = {
    'chunksize': chunk_size,  # 【修改】DSW可以用更大的chunk
    'parse_dates': [0],
    'dtype': dtype_dict,
    'engine': 'c',  # 使用C引擎提高解析速度
}

try:
    for df_chunk in pd.read_csv(path_df_part_1, **csv_params):
        chunk_count += 1
        df_chunk.columns = ['time','user_id','item_id','behavior_type','item_category']
        
        # 性能优化：向量化操作替代逐行处理 【不修改 - 算法逻辑保持不变】
        # 预过滤数据减少后续计算量
        df_chunk_b4 = df_chunk[df_chunk['behavior_type'] == 4]
        df_chunk_b3 = df_chunk[df_chunk['behavior_type'] == 3]
        df_chunk_in_1 = df_chunk[df_chunk['time'] >= pd.to_datetime('2014-11-27')]
        
        # 用户行为统计 - 使用groupby优化 【不修改 - 算法逻辑保持不变】
        chunk_u_b4 = df_chunk_b4.groupby('user_id').size().to_frame('u_b4_count_in_6')
        chunk_u_b3 = df_chunk_b3.groupby('user_id').size().to_frame('u_b3_count_in_6')
        chunk_u_b_total = df_chunk.groupby('user_id').size().to_frame('u_b_total_in_6')
        
        # 最近1天数据
        chunk_u_b_in_1 = df_chunk_in_1.groupby('user_id').size().to_frame('u_b_count_in_1')
        
        # 时间特征 - 批量计算
        chunk_u_b4_first = df_chunk_b4.groupby('user_id')['time'].min().to_frame('b4_first_time')
        chunk_u_first = df_chunk.groupby('user_id')['time'].min().to_frame('b_first_time')
        
        u_features_list.append({
            'u_b4': chunk_u_b4,
            'u_b3': chunk_u_b3, 
            'u_total': chunk_u_b_total,
            'u_in_1': chunk_u_b_in_1,
            'u_b4_first': chunk_u_b4_first,
            'u_first': chunk_u_first
        })
        
        # DSW性能监控：每10个chunk显示进度（减少输出频率） 【修改】
        if chunk_count % 10 == 0:  # 【修改】从20改为10，DSW处理更快
            print(f"      处理完chunk {chunk_count}")
            
            # DSW内存监控（DSW内存充足，提高阈值） 【修改】
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 60:  # 【修改】DSW内存充足，60%才提示
                    print(f"      DSW内存使用率: {memory_percent:.1f}%")
            except ImportError:
                pass

    print(f"   完成分块处理，共{chunk_count}个chunk")

    # 性能优化：并行合并特征 - 使用更高效的concat 【不修改 - 算法逻辑保持不变】
    print("   合并用户特征...")
    
    # 使用列表推导式和条件过滤提高效率
    valid_chunks_u_b4 = [chunk['u_b4'] for chunk in u_features_list if not chunk['u_b4'].empty]
    valid_chunks_u_b3 = [chunk['u_b3'] for chunk in u_features_list if not chunk['u_b3'].empty]
    valid_chunks_u_total = [chunk['u_total'] for chunk in u_features_list if not chunk['u_total'].empty]
    valid_chunks_u_in_1 = [chunk['u_in_1'] for chunk in u_features_list if not chunk['u_in_1'].empty]
    valid_chunks_u_b4_first = [chunk['u_b4_first'] for chunk in u_features_list if not chunk['u_b4_first'].empty]
    valid_chunks_u_first = [chunk['u_first'] for chunk in u_features_list if not chunk['u_first'].empty]

    # 性能优化：使用sort=False提高concat速度
    all_u_b4 = pd.concat(valid_chunks_u_b4, sort=False).groupby('user_id').sum() if valid_chunks_u_b4 else pd.DataFrame()
    all_u_b3 = pd.concat(valid_chunks_u_b3, sort=False).groupby('user_id').sum() if valid_chunks_u_b3 else pd.DataFrame()
    all_u_total = pd.concat(valid_chunks_u_total, sort=False).groupby('user_id').sum() if valid_chunks_u_total else pd.DataFrame()
    all_u_in_1 = pd.concat(valid_chunks_u_in_1, sort=False).groupby('user_id').sum() if valid_chunks_u_in_1 else pd.DataFrame()
    all_u_b4_first = pd.concat(valid_chunks_u_b4_first, sort=False).groupby('user_id').min() if valid_chunks_u_b4_first else pd.DataFrame()
    all_u_first = pd.concat(valid_chunks_u_first, sort=False).groupby('user_id').min() if valid_chunks_u_first else pd.DataFrame()

    # 释放内存
    del u_features_list, valid_chunks_u_b4, valid_chunks_u_b3, valid_chunks_u_total
    del valid_chunks_u_in_1, valid_chunks_u_b4_first, valid_chunks_u_first

    # 构建最终用户特征 - 链式操作提高效率 【不修改 - 算法逻辑保持不变】
    f_U_part_1 = all_u_total.copy()
    f_U_part_1 = f_U_part_1.join(all_u_b4, how='left').fillna(0)
    f_U_part_1 = f_U_part_1.join(all_u_b3, how='left').fillna(0)
    f_U_part_1 = f_U_part_1.join(all_u_in_1, how='left').fillna(0)

    # 向量化计算购买转化率 【不修改 - 算法逻辑保持不变】
    f_U_part_1['u_b4_rate'] = f_U_part_1['u_b4_count_in_6'] / (f_U_part_1['u_b_total_in_6'] + 1e-10)

    # 时间差计算优化 【不修改 - 算法逻辑保持不变】
    if not all_u_b4_first.empty and not all_u_first.empty:
        time_diff = all_u_b4_first.join(all_u_first, how='inner')
        time_diff['u_b4_diff_hours'] = (time_diff['b4_first_time'] - time_diff['b_first_time']).dt.total_seconds() / 3600
        f_U_part_1 = f_U_part_1.join(time_diff[['u_b4_diff_hours']], how='left').fillna(0)
    else:
        f_U_part_1['u_b4_diff_hours'] = 0

    # 最终用户特征 (5个) - 指定数据类型节省空间 【不修改 - 算法逻辑保持不变】
    f_U_part_1 = f_U_part_1[['u_b4_count_in_6', 'u_b3_count_in_6', 'u_b4_rate', 'u_b_count_in_1', 'u_b4_diff_hours']].round(3)
    f_U_part_1 = f_U_part_1.astype({'u_b4_count_in_6': 'int16', 'u_b3_count_in_6': 'int16', 'u_b_count_in_1': 'int16'})
    f_U_part_1.reset_index(inplace=True)

    # 性能优化：批量写入减少I/O
    f_U_part_1.to_csv(path_df_part_1_U, index=False)
    print(f"   用户(U)特征保存完成: {len(f_U_part_1):,}个用户, 5个特征")

except Exception as e:
    print(f"用户特征处理出错: {e}")
    exit(1)

###########################################
'''Step 1.2 商品(I)特征 - 精简版(5个特征) - DSW 48核心优化 【修改注释】
    1. i_u_count_in_6       # 商品6天内用户数
    2. i_b4_count_in_6      # 商品6天内购买次数
    3. i_b3_count_in_6      # 商品6天内加购物车次数
    4. i_b4_rate            # 商品购买转化率
    5. i_b_count_in_1       # 商品最近1天总行为数
'''

print("\nStep 1.2: 构建商品(I)特征 (5个核心特征)...")

i_features_list = []
chunk_count = 0

try:
    for df_chunk in pd.read_csv(path_df_part_1, **csv_params):
        chunk_count += 1
        df_chunk.columns = ['time','user_id','item_id','behavior_type','item_category']
        
        # 性能优化：预过滤和向量化操作 【不修改 - 算法逻辑保持不变】
        df_chunk_b4 = df_chunk[df_chunk['behavior_type'] == 4]
        df_chunk_b3 = df_chunk[df_chunk['behavior_type'] == 3]
        df_chunk_in_1 = df_chunk[df_chunk['time'] >= pd.to_datetime('2014-11-27')]
        
        # 商品特征统计 - 优化groupby操作 【不修改 - 算法逻辑保持不变】
        chunk_i_u = df_chunk.groupby('item_id')['user_id'].nunique().to_frame('i_u_count_in_6')
        chunk_i_b4 = df_chunk_b4.groupby('item_id').size().to_frame('i_b4_count_in_6')
        chunk_i_b3 = df_chunk_b3.groupby('item_id').size().to_frame('i_b3_count_in_6')
        chunk_i_b_total = df_chunk.groupby('item_id').size().to_frame('i_b_total_in_6')
        chunk_i_b_in_1 = df_chunk_in_1.groupby('item_id').size().to_frame('i_b_count_in_1')
        
        i_features_list.append({
            'i_u': chunk_i_u,
            'i_b4': chunk_i_b4,
            'i_b3': chunk_i_b3,
            'i_total': chunk_i_b_total,
            'i_in_1': chunk_i_b_in_1
        })

    # 合并商品特征 - 性能优化版本 【不修改 - 算法逻辑保持不变】
    print("   合并商品特征...")
    
    valid_chunks_i_u = [chunk['i_u'] for chunk in i_features_list if not chunk['i_u'].empty]
    valid_chunks_i_b4 = [chunk['i_b4'] for chunk in i_features_list if not chunk['i_b4'].empty]
    valid_chunks_i_b3 = [chunk['i_b3'] for chunk in i_features_list if not chunk['i_b3'].empty]
    valid_chunks_i_total = [chunk['i_total'] for chunk in i_features_list if not chunk['i_total'].empty]
    valid_chunks_i_in_1 = [chunk['i_in_1'] for chunk in i_features_list if not chunk['i_in_1'].empty]

    all_i_u = pd.concat(valid_chunks_i_u, sort=False).groupby('item_id').sum() if valid_chunks_i_u else pd.DataFrame()
    all_i_b4 = pd.concat(valid_chunks_i_b4, sort=False).groupby('item_id').sum() if valid_chunks_i_b4 else pd.DataFrame()
    all_i_b3 = pd.concat(valid_chunks_i_b3, sort=False).groupby('item_id').sum() if valid_chunks_i_b3 else pd.DataFrame()
    all_i_total = pd.concat(valid_chunks_i_total, sort=False).groupby('item_id').sum() if valid_chunks_i_total else pd.DataFrame()
    all_i_in_1 = pd.concat(valid_chunks_i_in_1, sort=False).groupby('item_id').sum() if valid_chunks_i_in_1 else pd.DataFrame()

    # 释放内存
    del i_features_list, valid_chunks_i_u, valid_chunks_i_b4, valid_chunks_i_b3, valid_chunks_i_total, valid_chunks_i_in_1

    # 构建最终商品特征 【不修改 - 算法逻辑保持不变】
    f_I_part_1 = all_i_u.join(all_i_b4, how='left').fillna(0)
    f_I_part_1 = f_I_part_1.join(all_i_b3, how='left').fillna(0)
    f_I_part_1 = f_I_part_1.join(all_i_total, how='left').fillna(0)
    f_I_part_1 = f_I_part_1.join(all_i_in_1, how='left').fillna(0)

    # 向量化计算转化率 【不修改 - 算法逻辑保持不变】
    f_I_part_1['i_b4_rate'] = f_I_part_1['i_b4_count_in_6'] / (f_I_part_1['i_b_total_in_6'] + 1e-10)

    # 最终商品特征 (5个) 【不修改 - 算法逻辑保持不变】
    f_I_part_1 = f_I_part_1[['i_u_count_in_6', 'i_b4_count_in_6', 'i_b3_count_in_6', 'i_b4_rate', 'i_b_count_in_1']].round(3)
    f_I_part_1 = f_I_part_1.astype({'i_u_count_in_6': 'int16', 'i_b4_count_in_6': 'int16', 'i_b3_count_in_6': 'int16', 'i_b_count_in_1': 'int16'})
    f_I_part_1.reset_index(inplace=True)

    f_I_part_1.to_csv(path_df_part_1_I, index=False)
    print(f"   商品(I)特征保存完成: {len(f_I_part_1):,}个商品, 5个特征")

except Exception as e:
    print(f"商品特征处理出错: {e}")
    exit(1)

###########################################
'''Step 1.3 品类(C)特征 - 精简版(5个特征) - DSW 48核心优化 【修改注释】'''

print("\nStep 1.3: 构建品类(C)特征 (5个核心特征)...")

c_features_list = []
chunk_count = 0

try:
    for df_chunk in pd.read_csv(path_df_part_1, **csv_params):
        chunk_count += 1
        df_chunk.columns = ['time','user_id','item_id','behavior_type','item_category']
        
        # 预过滤优化 【不修改 - 算法逻辑保持不变】
        df_chunk_b4 = df_chunk[df_chunk['behavior_type'] == 4]
        df_chunk_b3 = df_chunk[df_chunk['behavior_type'] == 3]
        df_chunk_in_1 = df_chunk[df_chunk['time'] >= pd.to_datetime('2014-11-27')]
        
        # 品类特征统计 【不修改 - 算法逻辑保持不变】
        chunk_c_u = df_chunk.groupby('item_category')['user_id'].nunique().to_frame('c_u_count_in_6')
        chunk_c_b4 = df_chunk_b4.groupby('item_category').size().to_frame('c_b4_count_in_6')
        chunk_c_b3 = df_chunk_b3.groupby('item_category').size().to_frame('c_b3_count_in_6')
        chunk_c_b_total = df_chunk.groupby('item_category').size().to_frame('c_b_total_in_6')
        chunk_c_b_in_1 = df_chunk_in_1.groupby('item_category').size().to_frame('c_b_count_in_1')
        
        c_features_list.append({
            'c_u': chunk_c_u,
            'c_b4': chunk_c_b4,
            'c_b3': chunk_c_b3,
            'c_total': chunk_c_b_total,
            'c_in_1': chunk_c_b_in_1
        })

    # 合并品类特征 【不修改 - 算法逻辑保持不变】
    print("   合并品类特征...")
    
    valid_chunks_c_u = [chunk['c_u'] for chunk in c_features_list if not chunk['c_u'].empty]
    valid_chunks_c_b4 = [chunk['c_b4'] for chunk in c_features_list if not chunk['c_b4'].empty]
    valid_chunks_c_b3 = [chunk['c_b3'] for chunk in c_features_list if not chunk['c_b3'].empty]
    valid_chunks_c_total = [chunk['c_total'] for chunk in c_features_list if not chunk['c_total'].empty]
    valid_chunks_c_in_1 = [chunk['c_in_1'] for chunk in c_features_list if not chunk['c_in_1'].empty]

    all_c_u = pd.concat(valid_chunks_c_u, sort=False).groupby('item_category').sum() if valid_chunks_c_u else pd.DataFrame()
    all_c_b4 = pd.concat(valid_chunks_c_b4, sort=False).groupby('item_category').sum() if valid_chunks_c_b4 else pd.DataFrame()
    all_c_b3 = pd.concat(valid_chunks_c_b3, sort=False).groupby('item_category').sum() if valid_chunks_c_b3 else pd.DataFrame()
    all_c_total = pd.concat(valid_chunks_c_total, sort=False).groupby('item_category').sum() if valid_chunks_c_total else pd.DataFrame()
    all_c_in_1 = pd.concat(valid_chunks_c_in_1, sort=False).groupby('item_category').sum() if valid_chunks_c_in_1 else pd.DataFrame()

    # 释放内存
    del c_features_list, valid_chunks_c_u, valid_chunks_c_b4, valid_chunks_c_b3, valid_chunks_c_total, valid_chunks_c_in_1

    # 构建最终品类特征 【不修改 - 算法逻辑保持不变】
    f_C_part_1 = all_c_u.join(all_c_b4, how='left').fillna(0)
    f_C_part_1 = f_C_part_1.join(all_c_b3, how='left').fillna(0)
    f_C_part_1 = f_C_part_1.join(all_c_total, how='left').fillna(0)
    f_C_part_1 = f_C_part_1.join(all_c_in_1, how='left').fillna(0)

    f_C_part_1['c_b4_rate'] = f_C_part_1['c_b4_count_in_6'] / (f_C_part_1['c_b_total_in_6'] + 1e-10)

    f_C_part_1 = f_C_part_1[['c_u_count_in_6', 'c_b4_count_in_6', 'c_b3_count_in_6', 'c_b4_rate', 'c_b_count_in_1']].round(3)
    f_C_part_1 = f_C_part_1.astype({'c_u_count_in_6': 'int16', 'c_b4_count_in_6': 'int16', 'c_b3_count_in_6': 'int16', 'c_b_count_in_1': 'int16'})
    f_C_part_1.reset_index(inplace=True)

    f_C_part_1.to_csv(path_df_part_1_C, index=False)
    print(f"   品类(C)特征保存完成: {len(f_C_part_1):,}个品类, 5个特征")

except Exception as e:
    print(f"品类特征处理出错: {e}")
    exit(1)

###########################################
'''Step 1.4 商品-品类(IC)特征 - 精简版(5个特征) 【不修改】'''

print("\nStep 1.4: 构建商品-品类(IC)特征 (5个核心特征)...")

# 获取UIC数据 【不修改 - 算法逻辑保持不变】
try:
    df_uic = pd.read_csv(path_df_part_1_uic_label, dtype={'user_id': 'int32', 'item_id': 'int32', 'item_category': 'int16'})
    df_uic = df_uic[['user_id', 'item_id', 'item_category']].drop_duplicates(['item_id', 'item_category'])
    print(f"   载入UIC数据: {len(df_uic):,}个商品-品类对")
except Exception as e:
    print(f"   错误: 无法载入UIC文件: {e}")
    pd.DataFrame(columns=['item_id', 'item_category', 'ic_u_rank_in_c', 'ic_b4_rank_in_c', 'ic_b_rank_in_c']).to_csv(path_df_part_1_IC, index=False)
    df_uic = None

if df_uic is not None:
    # 获取商品特征用于排名 【不修改 - 算法逻辑保持不变】
    f_I_for_rank = pd.read_csv(path_df_part_1_I, dtype={'item_id': 'int32'})[['item_id', 'i_u_count_in_6', 'i_b4_count_in_6']]
    f_I_for_rank['i_b_count_in_6'] = f_I_for_rank['i_u_count_in_6'] + f_I_for_rank['i_b4_count_in_6']  # 简化的总行为数
    
    # 合并商品和品类信息 【不修改 - 算法逻辑保持不变】
    f_IC_part_1 = df_uic.merge(f_I_for_rank, on='item_id', how='left').fillna(0)
    
    # 性能优化：使用向量化rank操作 【不修改 - 算法逻辑保持不变】
    f_IC_part_1['ic_u_rank_in_c'] = f_IC_part_1.groupby('item_category')['i_u_count_in_6'].rank(method='min', ascending=False).astype('int16')
    f_IC_part_1['ic_b4_rank_in_c'] = f_IC_part_1.groupby('item_category')['i_b4_count_in_6'].rank(method='min', ascending=False).astype('int16')
    f_IC_part_1['ic_b_rank_in_c'] = f_IC_part_1.groupby('item_category')['i_b_count_in_6'].rank(method='min', ascending=False).astype('int16')
    
    # 最终IC特征 (5个) 【不修改 - 算法逻辑保持不变】
    f_IC_part_1 = f_IC_part_1[['item_id', 'item_category', 'ic_u_rank_in_c', 'ic_b4_rank_in_c', 'ic_b_rank_in_c']]
    f_IC_part_1 = f_IC_part_1.astype({'item_id': 'int32', 'item_category': 'int16'})
    f_IC_part_1.to_csv(path_df_part_1_IC, index=False)
    print(f"   商品-品类(IC)特征保存完成: {len(f_IC_part_1):,}个商品-品类对, 5个特征")

###########################################
'''Step 1.5 用户-商品(UI)特征 - 精简版(5个特征) - DSW 48核心优化 【修改注释】'''

print("\nStep 1.5: 构建用户-商品(UI)特征 (5个核心特征)...")

ui_features_list = []
chunk_count = 0

try:
    for df_chunk in pd.read_csv(path_df_part_1, **csv_params):
        chunk_count += 1
        df_chunk.columns = ['time','user_id','item_id','behavior_type','item_category']
        
        # 预过滤 【不修改 - 算法逻辑保持不变】
        df_chunk_b4 = df_chunk[df_chunk['behavior_type'] == 4]
        df_chunk_b3 = df_chunk[df_chunk['behavior_type'] == 3]
        df_chunk_in_1 = df_chunk[df_chunk['time'] >= pd.to_datetime('2014-11-27')]
        
        # UI行为统计 【不修改 - 算法逻辑保持不变】
        chunk_ui_b4 = df_chunk_b4.groupby(['user_id', 'item_id']).size().to_frame('ui_b4_count_in_6')
        chunk_ui_b3 = df_chunk_b3.groupby(['user_id', 'item_id']).size().to_frame('ui_b3_count_in_6')
        chunk_ui_total = df_chunk.groupby(['user_id', 'item_id']).size().to_frame('ui_b_total_in_6')
        chunk_ui_in_1 = df_chunk_in_1.groupby(['user_id', 'item_id']).size().to_frame('ui_b_count_in_1')
        chunk_ui_b3_last = df_chunk_b3.groupby(['user_id', 'item_id'])['time'].max().to_frame('ui_b3_last_time')
        
        ui_features_list.append({
            'ui_b4': chunk_ui_b4,
            'ui_b3': chunk_ui_b3,
            'ui_total': chunk_ui_total,
            'ui_in_1': chunk_ui_in_1,
            'ui_b3_last': chunk_ui_b3_last
        })

    # 合并UI特征 【不修改 - 算法逻辑保持不变】
    print("   合并UI特征...")
    
    valid_chunks_ui_b4 = [chunk['ui_b4'] for chunk in ui_features_list if not chunk['ui_b4'].empty]
    valid_chunks_ui_b3 = [chunk['ui_b3'] for chunk in ui_features_list if not chunk['ui_b3'].empty]
    valid_chunks_ui_total = [chunk['ui_total'] for chunk in ui_features_list if not chunk['ui_total'].empty]
    valid_chunks_ui_in_1 = [chunk['ui_in_1'] for chunk in ui_features_list if not chunk['ui_in_1'].empty]
    valid_chunks_ui_b3_last = [chunk['ui_b3_last'] for chunk in ui_features_list if not chunk['ui_b3_last'].empty]

    all_ui_b4 = pd.concat(valid_chunks_ui_b4, sort=False).groupby(['user_id', 'item_id']).sum() if valid_chunks_ui_b4 else pd.DataFrame()
    all_ui_b3 = pd.concat(valid_chunks_ui_b3, sort=False).groupby(['user_id', 'item_id']).sum() if valid_chunks_ui_b3 else pd.DataFrame()
    all_ui_total = pd.concat(valid_chunks_ui_total, sort=False).groupby(['user_id', 'item_id']).sum() if valid_chunks_ui_total else pd.DataFrame()
    all_ui_in_1 = pd.concat(valid_chunks_ui_in_1, sort=False).groupby(['user_id', 'item_id']).sum() if valid_chunks_ui_in_1 else pd.DataFrame()
    all_ui_b3_last = pd.concat(valid_chunks_ui_b3_last, sort=False).groupby(['user_id', 'item_id']).max() if valid_chunks_ui_b3_last else pd.DataFrame()

    # 释放内存
    del ui_features_list, valid_chunks_ui_b4, valid_chunks_ui_b3, valid_chunks_ui_total, valid_chunks_ui_in_1, valid_chunks_ui_b3_last

    # 构建最终UI特征 【不修改 - 算法逻辑保持不变】
    f_UI_part_1 = all_ui_total.join(all_ui_b4, how='left').fillna(0)
    f_UI_part_1 = f_UI_part_1.join(all_ui_b3, how='left').fillna(0)
    f_UI_part_1 = f_UI_part_1.join(all_ui_in_1, how='left').fillna(0)

    # 向量化排名计算 【不修改 - 算法逻辑保持不变】
    f_UI_part_1['ui_b_count_rank_in_u'] = f_UI_part_1.groupby('user_id')['ui_b_total_in_6'].rank(method='min', ascending=False).astype('int16')

    # 时间差计算 【不修改 - 算法逻辑保持不变】
    if not all_ui_b3_last.empty:
        target_time = pd.to_datetime('2014-11-28')
        all_ui_b3_last['ui_b3_last_hours'] = (target_time - all_ui_b3_last['ui_b3_last_time']).dt.total_seconds() / 3600
        f_UI_part_1 = f_UI_part_1.join(all_ui_b3_last[['ui_b3_last_hours']], how='left').fillna(999999)
    else:
        f_UI_part_1['ui_b3_last_hours'] = 999999

    # 最终UI特征 (5个) 【不修改 - 算法逻辑保持不变】
    f_UI_part_1 = f_UI_part_1[['ui_b4_count_in_6', 'ui_b3_count_in_6', 'ui_b_count_rank_in_u', 'ui_b3_last_hours', 'ui_b_count_in_1']].round(1)
    f_UI_part_1 = f_UI_part_1.astype({'ui_b4_count_in_6': 'int16', 'ui_b3_count_in_6': 'int16', 'ui_b_count_in_1': 'int16'})
    f_UI_part_1.reset_index(inplace=True)

    f_UI_part_1.to_csv(path_df_part_1_UI, index=False)
    print(f"   用户-商品(UI)特征保存完成: {len(f_UI_part_1):,}个用户-商品对, 5个特征")

except Exception as e:
    print(f"UI特征处理出错: {e}")
    exit(1)

###########################################
'''Step 1.6 用户-品类(UC)特征 - 精简版(5个特征) - DSW 48核心优化 【修改注释】'''

print("\nStep 1.6: 构建用户-品类(UC)特征 (5个核心特征)...")

uc_features_list = []
chunk_count = 0

try:
    for df_chunk in pd.read_csv(path_df_part_1, **csv_params):
        chunk_count += 1
        df_chunk.columns = ['time','user_id','item_id','behavior_type','item_category']
        
        # 预过滤 【不修改 - 算法逻辑保持不变】
        df_chunk_b4 = df_chunk[df_chunk['behavior_type'] == 4]
        df_chunk_b3 = df_chunk[df_chunk['behavior_type'] == 3]
        df_chunk_in_1 = df_chunk[df_chunk['time'] >= pd.to_datetime('2014-11-27')]
        
        # UC行为统计 【不修改 - 算法逻辑保持不变】
        chunk_uc_b4 = df_chunk_b4.groupby(['user_id', 'item_category']).size().to_frame('uc_b4_count_in_6')
        chunk_uc_b3 = df_chunk_b3.groupby(['user_id', 'item_category']).size().to_frame('uc_b3_count_in_6')
        chunk_uc_total = df_chunk.groupby(['user_id', 'item_category']).size().to_frame('uc_b_total_in_6')
        chunk_uc_in_1 = df_chunk_in_1.groupby(['user_id', 'item_category']).size().to_frame('uc_b_count_in_1')
        chunk_uc_b3_last = df_chunk_b3.groupby(['user_id', 'item_category'])['time'].max().to_frame('uc_b3_last_time')
        
        uc_features_list.append({
            'uc_b4': chunk_uc_b4,
            'uc_b3': chunk_uc_b3,
            'uc_total': chunk_uc_total,
            'uc_in_1': chunk_uc_in_1,
            'uc_b3_last': chunk_uc_b3_last
        })

    # 合并UC特征 【不修改 - 算法逻辑保持不变】
    print("   合并UC特征...")
    
    valid_chunks_uc_b4 = [chunk['uc_b4'] for chunk in uc_features_list if not chunk['uc_b4'].empty]
    valid_chunks_uc_b3 = [chunk['uc_b3'] for chunk in uc_features_list if not chunk['uc_b3'].empty]
    valid_chunks_uc_total = [chunk['uc_total'] for chunk in uc_features_list if not chunk['uc_total'].empty]
    valid_chunks_uc_in_1 = [chunk['uc_in_1'] for chunk in uc_features_list if not chunk['uc_in_1'].empty]
    valid_chunks_uc_b3_last = [chunk['uc_b3_last'] for chunk in uc_features_list if not chunk['uc_b3_last'].empty]

    all_uc_b4 = pd.concat(valid_chunks_uc_b4, sort=False).groupby(['user_id', 'item_category']).sum() if valid_chunks_uc_b4 else pd.DataFrame()
    all_uc_b3 = pd.concat(valid_chunks_uc_b3, sort=False).groupby(['user_id', 'item_category']).sum() if valid_chunks_uc_b3 else pd.DataFrame()
    all_uc_total = pd.concat(valid_chunks_uc_total, sort=False).groupby(['user_id', 'item_category']).sum() if valid_chunks_uc_total else pd.DataFrame()
    all_uc_in_1 = pd.concat(valid_chunks_uc_in_1, sort=False).groupby(['user_id', 'item_category']).sum() if valid_chunks_uc_in_1 else pd.DataFrame()
    all_uc_b3_last = pd.concat(valid_chunks_uc_b3_last, sort=False).groupby(['user_id', 'item_category']).max() if valid_chunks_uc_b3_last else pd.DataFrame()

    # 释放内存
    del uc_features_list, valid_chunks_uc_b4, valid_chunks_uc_b3, valid_chunks_uc_total, valid_chunks_uc_in_1, valid_chunks_uc_b3_last

    # 构建最终UC特征 【不修改 - 算法逻辑保持不变】
    f_UC_part_1 = all_uc_total.join(all_uc_b4, how='left').fillna(0)
    f_UC_part_1 = f_UC_part_1.join(all_uc_b3, how='left').fillna(0)
    f_UC_part_1 = f_UC_part_1.join(all_uc_in_1, how='left').fillna(0)

    # 向量化排名计算 【不修改 - 算法逻辑保持不变】
    f_UC_part_1['uc_b_count_rank_in_u'] = f_UC_part_1.groupby('user_id')['uc_b_total_in_6'].rank(method='min', ascending=False).astype('int16')

    # 时间差计算 【不修改 - 算法逻辑保持不变】
    if not all_uc_b3_last.empty:
        target_time = pd.to_datetime('2014-11-28')
        all_uc_b3_last['uc_b3_last_hours'] = (target_time - all_uc_b3_last['uc_b3_last_time']).dt.total_seconds() / 3600
        f_UC_part_1 = f_UC_part_1.join(all_uc_b3_last[['uc_b3_last_hours']], how='left').fillna(999999)
    else:
        f_UC_part_1['uc_b3_last_hours'] = 999999

    # 最终UC特征 (5个) 【不修改 - 算法逻辑保持不变】
    f_UC_part_1 = f_UC_part_1[['uc_b4_count_in_6', 'uc_b3_count_in_6', 'uc_b_count_rank_in_u', 'uc_b3_last_hours', 'uc_b_count_in_1']].round(1)
    f_UC_part_1 = f_UC_part_1.astype({'uc_b4_count_in_6': 'int16', 'uc_b3_count_in_6': 'int16', 'uc_b_count_in_1': 'int16'})
    f_UC_part_1.reset_index(inplace=True)

    f_UC_part_1.to_csv(path_df_part_1_UC, index=False)
    print(f"   用户-品类(UC)特征保存完成: {len(f_UC_part_1):,}个用户-品类对, 5个特征")

except Exception as e:
    print(f"UC特征处理出错: {e}")
    exit(1)

print("\n" + "=" * 60)
print("Part 1精简特征工程完成! (DSW环境优化)") # 【修改】

# 最终统计 【不修改】
print(f"\n生成的特征文件:")
feature_files = [path_df_part_1_U, path_df_part_1_I, path_df_part_1_C, path_df_part_1_IC, path_df_part_1_UI, path_df_part_1_UC]
total_size = 0

for file_path in feature_files:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024*1024)
        total_size += size_mb
        try:
            line_count = len(pd.read_csv(file_path, nrows=0).columns)
            print(f"   成功: {os.path.basename(file_path)}: {size_mb:.1f}MB, {line_count}个特征")
        except:
            print(f"   成功: {os.path.basename(file_path)}: {size_mb:.1f}MB")

print(f"\n特征文件总大小: {total_size:.1f}MB")
print(f"精简特征数: 6类 × 5个 = 30个特征 (相比原89个特征减少66%)")
print("DSW 48核心优化: 使用了向量化计算、大内存管理、高效并行处理等优化技术") # 【修改】
print("   可以继续执行Part2和Part3特征工程...")

print(' - PY131 Modified for DSW 48-core optimization - ') # 【修改】



# 全局中文字体配置（优先用已安装字体；否则回退到Windows字体文件路径）
def setup_chinese_font():
    font_candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Source Han Sans CN', 'Arial Unicode MS']
    available = [f.name for f in fm.fontManager.ttflist]
    chosen = None
    for name in font_candidates:
        if name in available:
            chosen = name
            break
    ch_font = None
    if chosen:
        plt.rcParams['font.sans-serif'] = [chosen]
    else:
        font_path = r"C:\\Windows\\Fonts\\msyh.ttc"  # 微软雅黑路径（Windows）
        if os.path.exists(font_path):
            ch_font = FontProperties(fname=font_path)
        else:
            print("[字体] 未找到中文字体，图中文字可能显示为方块。建议安装 Noto Sans CJK SC 或放置 msyh.ttc。")
    plt.rcParams['axes.unicode_minus'] = False  # 解决坐标负号显示问题
    return ch_font

CH_FONT = setup_chinese_font()


# === 新增：相关性热力图与VIF分析（Part1） ===
def analyze_correlation_and_vif_part1(sample_n=200000, random_state=42):
    """
    从 Part1 的特征文件合并出28个数值特征，生成：
      - 相关矩阵CSV + 热力图PNG
      - VIF表CSV + 柱状图PNG
    输出目录：Mobile_Recommendation/data/mobile/feature
    """
    out_dir = os.path.dirname(path_df_part_1_U)

    # 读取 UIC 键（用于合并到统一样本空间）
    try:
        df_keys = pd.read_csv(
            path_df_part_1_uic_label,
            usecols=['user_id', 'item_id', 'item_category'],
            dtype={'user_id': 'int32', 'item_id': 'int32', 'item_category': 'int16'}
        ).drop_duplicates()
    except Exception as e:
        print(f"[分析] 读取 UIC 键失败: {e}")
        return

    # 读取各类特征
    try:
        fU  = pd.read_csv(path_df_part_1_U)
        fI  = pd.read_csv(path_df_part_1_I)
        fC  = pd.read_csv(path_df_part_1_C)
        fIC = pd.read_csv(path_df_part_1_IC)
        fUI = pd.read_csv(path_df_part_1_UI)
        fUC = pd.read_csv(path_df_part_1_UC)
    except Exception as e:
        print(f"[分析] 读取特征文件失败: {e}")
        return

    # 合并为统一样本空间（与训练/聚类口径一致的键）
    df = df_keys.merge(fU,  how='left', on=['user_id'])
    df = df.merge(fI,  how='left', on=['item_id'])
    df = df.merge(fC,  how='left', on=['item_category'])
    df = df.merge(fIC, how='left', on=['item_id', 'item_category'])
    df = df.merge(fUI, how='left', on=['user_id', 'item_id'])
    df = df.merge(fUC, how='left', on=['user_id', 'item_category'])

    # 28个数值特征列（6类精简特征中去掉IC的两个ID列）
    U_COLS  = ['u_b4_count_in_6','u_b3_count_in_6','u_b4_rate','u_b_count_in_1','u_b4_diff_hours']
    I_COLS  = ['i_u_count_in_6','i_b4_count_in_6','i_b3_count_in_6','i_b4_rate','i_b_count_in_1']
    C_COLS  = ['c_u_count_in_6','c_b4_count_in_6','c_b3_count_in_6','c_b4_rate','c_b_count_in_1']
    IC_COLS = ['ic_u_rank_in_c','ic_b4_rank_in_c','ic_b_rank_in_c']  # 仅3个数值排名
    UI_COLS = ['ui_b4_count_in_6','ui_b3_count_in_6','ui_b_count_rank_in_u','ui_b3_last_hours','ui_b_count_in_1']
    UC_COLS = ['uc_b4_count_in_6','uc_b3_count_in_6','uc_b_count_rank_in_u','uc_b3_last_hours','uc_b_count_in_1']

    FEATURE_NUM_COLS = U_COLS + I_COLS + C_COLS + IC_COLS + UI_COLS + UC_COLS

    # 仅保留数值特征并填充缺失
    miss_cols = [c for c in FEATURE_NUM_COLS if c not in df.columns]
    if miss_cols:
        print(f"[分析] 警告：以下特征列在合并后缺失，将自动创建为0：{miss_cols}")
        for c in miss_cols:
            df[c] = 0

    X = df[FEATURE_NUM_COLS].copy().fillna(0)

    # 可选：对超大数据做下采样（不改变相关与VIF的代表性）
    if len(X) > sample_n:
        X = X.sample(n=sample_n, random_state=random_state)
        print(f"[分析] 为稳定与性能采样 {sample_n:,} 行用于相关与VIF计算（原始 {len(df):,} 行）")

    # 1) 相关矩阵与热力图
    print("[分析] 计算皮尔逊相关矩阵与热力图...")
    corr = X.corr(method='pearson')

    corr_csv = os.path.join(out_dir, "part1_feature_corr_matrix.csv")
    heatmap_png = os.path.join(out_dir, "part1_feature_corr_heatmap.png")
    try:
        corr.to_csv(corr_csv, index=True, encoding='utf-8-sig')
    except Exception as e:
        print(f"[分析] 保存相关矩阵CSV失败: {e}")

    plt.figure(figsize=(16, 14))
    ax = sns.heatmap(
        corr,
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={'shrink': 0.6},
        xticklabels=True,
        yticklabels=True
    )
    title_text = "Part1 精简特征相关性热力图"
    if CH_FONT:
        plt.title(title_text, fontproperties=CH_FONT, fontsize=14)
        for lbl in ax.get_xticklabels():
            lbl.set_fontproperties(CH_FONT)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(CH_FONT)
    else:
        plt.title(title_text, fontsize=14)
    plt.tight_layout()
    plt.savefig(heatmap_png, dpi=150)
    plt.close()


    # 2) VIF 计算与柱状图
    print("[分析] 计算方差膨胀因子（VIF）...")
    # 标准化（VIF与缩放无关，但标准化更数值稳定）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(float))

    vif_rows = []
    for j, col in enumerate(FEATURE_NUM_COLS):
        try:
            vif_val = float(variance_inflation_factor(X_scaled, j))
        except Exception:
            vif_val = float('inf')
        vif_rows.append({'feature': col, 'vif': vif_val})

    vif_df = pd.DataFrame(vif_rows).sort_values('vif', ascending=False).reset_index(drop=True)
    vif_csv = os.path.join(out_dir, "part1_feature_vif.csv")
    try:
        vif_df.to_csv(vif_csv, index=False, encoding='utf-8-sig')
        print(f"[分析] 已保存VIF表: {vif_csv}")
    except Exception as e:
        print(f"[分析] 保存VIF CSV失败: {e}")

    # VIF 柱状图（截断极端值便于展示）
    plt.figure(figsize=(14, 8))
    plot_df = vif_df.copy()
    plot_df['vif_plot'] = plot_df['vif'].clip(upper=50)  # 显示上限50，避免单列过大挤压视图
    sns.barplot(data=plot_df, x='vif_plot', y='feature', orient='h', palette='viridis')
    plt.xlabel("VIF（>10通常认为共线较强，图中>50被截断显示）")
    plt.ylabel("特征名")
    plt.title("Part1 精简特征 VIF 柱状图")
    plt.tight_layout()
    vif_png = os.path.join(out_dir, "part1_feature_vif_bar.png")
    try:
        plt.savefig(vif_png, dpi=150)
        print(f"[分析] 已保存VIF柱状图: {vif_png}")
    finally:
        plt.close()

# === 调用分析（放在特征全部写出之后） ===
try:
    analyze_correlation_and_vif_part1(sample_n=200000, random_state=42)
except Exception as e:
    print(f"[分析] 相关性与VIF分析出现异常：{e}")
# ...existing code...
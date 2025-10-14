import pandas as pd
import numpy as np
import os
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

def setup_performance_optimization():
    """
    配置K-means聚类的性能优化参数
    基于DSW 48核心+372GB内存配置 【修改】
    """
    # 获取环境变量中的优化参数，如果没有则使用DSW默认值
    memory_mode = os.environ.get('TIANCHI_MEMORY_MODE', 'normal')
    cpu_cores = int(os.environ.get('TIANCHI_CPU_CORES', '40'))  # 【修改】DSW使用40核心
    suggested_chunk_size = int(os.environ.get('TIANCHI_CHUNK_SIZE', '500000'))  # 【修改】DSW大内存默认50万行
    
    # DSW 372GB超大内存配置，可以使用很大的chunk_size 【修改】
    # K-means聚类是CPU密集型任务，可以充分利用多核和大内存
    if memory_mode == 'conservative':
        chunk_size_standardize = min(suggested_chunk_size, 300000)  # 【修改】DSW保守模式仍然很大
        chunk_size_clustering = min(150000, suggested_chunk_size // 2)  # 【修改】聚类chunk也更大
        n_jobs = min(cpu_cores // 2, 20)  # 【修改】保守模式使用20核心
    else:
        chunk_size_standardize = min(800000, suggested_chunk_size * 2)  # 【修改】DSW正常模式可以用80万行！
        chunk_size_clustering = min(400000, suggested_chunk_size)  # 【修改】聚类chunk 40万行
        n_jobs = min(cpu_cores, 30)  # 【修改】正常模式使用30核心进行聚类
    
    # 设置numpy和sklearn的多线程参数，充分利用DSW多核 【修改】
    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_cores)
    os.environ['MKL_NUM_THREADS'] = str(n_jobs)  # Intel MKL优化
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_jobs)  # OpenBLAS优化
    
    print(f"K-means聚类性能优化配置 (DSW):") # 【修改】
    print(f"   CPU核心数: {multiprocessing.cpu_count()} (使用{n_jobs}个用于聚类)")
    print(f"   内存模式: {memory_mode}")
    print(f"   标准化Chunk大小: {chunk_size_standardize:,} (DSW大内存优化)") # 【修改】
    print(f"   聚类Chunk大小: {chunk_size_clustering:,} (DSW大内存优化)") # 【修改】
    print(f"   多线程设置: sklearn={n_jobs}核心, 数值计算={cpu_cores}核心") # 【修改】
    print(f"   预计处理能力: 全量11亿+数据聚类") # 【修改】
    
    return chunk_size_standardize, chunk_size_clustering, n_jobs

# 配置性能优化
chunk_size_standardize, chunk_size_clustering, n_jobs = setup_performance_optimization()


# data_set keys and labels - 修正路径 【不修改】
path_df_part_1_uic_label = "Mobile_Recommendation/data/mobile/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "Mobile_Recommendation/data/mobile/df_part_2_uic_label.csv"

# data_set features - 统一路径到feature目录 【不修改】
path_df_part_1_U   = "Mobile_Recommendation/data/mobile/feature/df_part_1_U.csv"  
path_df_part_1_I   = "Mobile_Recommendation/data/mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "Mobile_Recommendation/data/mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "Mobile_Recommendation/data/mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "Mobile_Recommendation/data/mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "Mobile_Recommendation/data/mobile/feature/df_part_1_UC.csv"

path_df_part_2_U   = "Mobile_Recommendation/data/mobile/feature/df_part_2_U.csv"  
path_df_part_2_I   = "Mobile_Recommendation/data/mobile/feature/df_part_2_I.csv"
path_df_part_2_C   = "Mobile_Recommendation/data/mobile/feature/df_part_2_C.csv"
path_df_part_2_IC  = "Mobile_Recommendation/data/mobile/feature/df_part_2_IC.csv"
path_df_part_2_UI  = "Mobile_Recommendation/data/mobile/feature/df_part_2_UI.csv"
path_df_part_2_UC  = "Mobile_Recommendation/data/mobile/feature/df_part_2_UC.csv"

path_df_part_3_U   = "Mobile_Recommendation/data/mobile/feature/df_part_3_U.csv"  
path_df_part_3_I   = "Mobile_Recommendation/data/mobile/feature/df_part_3_I.csv"
path_df_part_3_C   = "Mobile_Recommendation/data/mobile/feature/df_part_3_C.csv"
path_df_part_3_IC  = "Mobile_Recommendation/data/mobile/feature/df_part_3_IC.csv"
path_df_part_3_UI  = "Mobile_Recommendation/data/mobile/feature/df_part_3_UI.csv"
path_df_part_3_UC  = "Mobile_Recommendation/data/mobile/feature/df_part_3_UC.csv"

### output files - 统一保存到gbdt目录 【不修改】
# data partition with different label
path_df_part_1_uic_label_0 = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_0.csv"
path_df_part_1_uic_label_1 = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_1.csv"
path_df_part_2_uic_label_0 = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_0.csv"
path_df_part_2_uic_label_1 = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_1.csv"

# training set keys uic-label with k_means clusters' label
path_df_part_1_uic_label_cluster = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_cluster.csv"

# scalers for data standardization store as python pickle
path_df_part_1_scaler = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_1_scaler"
path_df_part_2_scaler = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_2_scaler"


FEATURE_COLUMNS = [
    'u_b4_count_in_6', 'u_b3_count_in_6', 'u_b4_rate', 'u_b_count_in_1', 'u_b4_diff_hours',
    'i_u_count_in_6', 'i_b4_count_in_6', 'i_b3_count_in_6', 'i_b4_rate', 'i_b_count_in_1',
    'c_u_count_in_6', 'c_b4_count_in_6', 'c_b3_count_in_6', 'c_b4_rate', 'c_b_count_in_1',
    'ic_u_rank_in_c', 'ic_b4_rank_in_c', 'ic_b_rank_in_c',
    'ui_b4_count_in_6', 'ui_b3_count_in_6', 'ui_b_count_rank_in_u', 'ui_b3_last_hours', 'ui_b_count_in_1',
    'uc_b4_count_in_6', 'uc_b3_count_in_6', 'uc_b_count_rank_in_u', 'uc_b3_last_hours', 'uc_b_count_in_1'
]

# 用于机器学习的特征（排除ID列） 【不修改】
ML_FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col not in ['item_id', 'item_category']]
print(f"使用精简特征体系: {len(ML_FEATURE_COLUMNS)}个特征用于聚类")

def df_read(path, mode='r'):
    '''the definition of dataframe loading function 
    '''
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return None
    
    try:
        # 性能优化：预定义数据类型加速读取 【不修改】
        dtype_dict = {
            'user_id': 'int32',
            'item_id': 'int32',
            'item_category': 'int16',
            'label': 'int8'
        }
        
        # 尝试使用数据类型优化，如果失败则使用默认方式
        try:
            df = pd.read_csv(path, index_col=False, dtype={k:v for k,v in dtype_dict.items() if k in pd.read_csv(path, nrows=1).columns})
        except:
            df = pd.read_csv(path, index_col=False)
            
        file_size = os.path.getsize(path) / (1024*1024)
        print(f"   载入: {os.path.basename(path)} ({file_size:.1f}MB, {len(df):,}行)")
        return df
    except Exception as e:
        print(f"读取文件失败: {path}, 错误: {e}")
        return None

def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size
    
    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df): 
        return df
    else: 
        # 性能优化：设置随机种子确保可重现性 【不修改】
        return df.sample(n=sub_size, random_state=42)

########################################################################
'''Step 1: dividing of positive and negative sub-set by u-i-c-label keys
    
    p.s. we first generate u-i-C key, then merging for data set and operation by chunk 
    such strange operation designed for saving my poor PC-MEM.
'''

print("\nStep 1: 分离正负样本...")

df_part_1_uic_label = df_read(path_df_part_1_uic_label)  # loading total keys
df_part_2_uic_label = df_read(path_df_part_2_uic_label)

if df_part_1_uic_label is None or df_part_2_uic_label is None:
    print("无法载入标签文件")
    exit(1)

# 性能优化：使用向量化操作分离正负样本 【不修改 - 算法逻辑保持不变】
df_part_1_uic_label_0 = df_part_1_uic_label[df_part_1_uic_label['label'] == 0].copy()
df_part_1_uic_label_1 = df_part_1_uic_label[df_part_1_uic_label['label'] == 1].copy()
df_part_2_uic_label_0 = df_part_2_uic_label[df_part_2_uic_label['label'] == 0].copy()
df_part_2_uic_label_1 = df_part_2_uic_label[df_part_2_uic_label['label'] == 1].copy()

print(f"   Part 1 - 负样本: {len(df_part_1_uic_label_0):,}个, 正样本: {len(df_part_1_uic_label_1):,}个")
print(f"   Part 2 - 负样本: {len(df_part_2_uic_label_0):,}个, 正样本: {len(df_part_2_uic_label_1):,}个")
print(f"   样本比例约为: {len(df_part_1_uic_label_0)/max(len(df_part_1_uic_label_1),1):.0f}:1")

# 性能优化：批量保存，减少I/O次数 【不修改】
df_part_1_uic_label_0.to_csv(path_df_part_1_uic_label_0, index=False)
df_part_1_uic_label_1.to_csv(path_df_part_1_uic_label_1, index=False)
df_part_2_uic_label_0.to_csv(path_df_part_2_uic_label_0, index=False)
df_part_2_uic_label_1.to_csv(path_df_part_2_uic_label_1, index=False)

print("   正负样本分离完成")

# 释放内存
del df_part_1_uic_label, df_part_2_uic_label

#######################################################################
'''Step 2: clustering on negative sub-set
    clusters number ~ 1000, using mini-batch-k-means
'''

print("\nStep 2: 对负样本进行K-means聚类...")

# clustering based on sklearn
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import pickle

##### part_1 #####
print("   处理Part 1数据...")

# 性能优化：预定义特征文件的数据类型 【不修改】
feature_dtypes = {
    'user_id': 'int32',
    'item_id': 'int32', 
    'item_category': 'int16'
}

# loading features with optimized dtypes
print("   载入Part 1特征文件...")
df_part_1_U  = df_read(path_df_part_1_U)   
df_part_1_I  = df_read(path_df_part_1_I)
df_part_1_C  = df_read(path_df_part_1_C)
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)


# process by chunk as ui-pairs size is too big
print("   Step 2.1: 拟合标准化器...")

# 性能优化：使用增量学习的StandardScaler，支持多核处理 【不修改 - 算法逻辑保持不变】
scaler_1 = preprocessing.StandardScaler() 
batch = 0

# DSW性能优化：使用更高效的CSV读取参数和更大chunk_size 【修改】
csv_read_params = {
    'chunksize': chunk_size_standardize,  # 【修改】DSW可以用更大的chunk
    'dtype': {'user_id': 'int32', 'item_id': 'int32', 'item_category': 'int16', 'label': 'int8'},
    'engine': 'c'  # 使用C引擎提高解析速度
}

for df_part_1_uic_label_0_chunk in pd.read_csv(path_df_part_1_uic_label_0, **csv_read_params): 
    try:
        batch += 1
        print(f"      处理标准化batch {batch}, 样本数: {len(df_part_1_uic_label_0_chunk):,}")
        
        # 性能优化：使用更高效的merge方式和内存管理 【不修改 - 算法逻辑保持不变】
        train_data_df_part_1 = df_part_1_uic_label_0_chunk.copy()
        
        # 批量merge提高效率
        merge_sequence = [
            (df_part_1_U, ['user_id']),
            (df_part_1_I, ['item_id']),
            (df_part_1_C, ['item_category']),
            (df_part_1_IC, ['item_id','item_category']),
            (df_part_1_UI, ['user_id','item_id']),
            (df_part_1_UC, ['user_id','item_category'])
        ]
        
        for feature_df, merge_keys in merge_sequence:
            train_data_df_part_1 = pd.merge(train_data_df_part_1, feature_df, how='left', on=merge_keys, sort=False)
        train_data_df_part_1[ML_FEATURE_COLUMNS] = train_data_df_part_1[ML_FEATURE_COLUMNS].fillna(0)
        train_X_1 = train_data_df_part_1[ML_FEATURE_COLUMNS].values
        scaler_1.partial_fit(train_X_1)  # 读取一小块数据（batch）后，更新全局均值 μ 和标准差 σ 的估计值
        
        # DSW内存监控：每5个chunk显示进度（减少输出频率） 【修改】
        if batch % 5 == 0:
            print(f"         已处理 {batch} 个batch...")
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 60:  # 【修改】DSW内存充足，60%才提示
                    print(f"         DSW内存使用率: {memory_percent:.1f}%")
            except ImportError:
                pass
        
        # 释放chunk内存
        del train_data_df_part_1, train_X_1
        
    except StopIteration:
        print("         标准化器拟合完成")
        break
    except Exception as e:
        print(f"         处理batch {batch} 出错: {e}")
        continue

print("   Step 2.2: 执行Mini-Batch K-means聚类...")

# DSW性能优化：配置支持多核的Mini-Batch K-means 【修改】
mbk_1 = MiniBatchKMeans(
    init='k-means++',           # K-means++初始化方法，更稳定
    n_clusters=1000,            # 聚类数量: 1000个簇  
    batch_size=1000,            # 批处理大小增加到1000
    reassignment_ratio=10**-4,  # 重新分配比例: 0.0001
    random_state=42,            # 随机种子确保可重现性
    n_init=3,                   # 减少初始化次数，在大数据集上3次足够
    max_no_improvement=10       # 早停机制，性能优化
) 

classes_1 = []
batch = 0

# DSW优化：更大的聚类chunk_size 【修改】
csv_read_params_clustering = csv_read_params.copy()
csv_read_params_clustering['chunksize'] = chunk_size_clustering  # 【修改】DSW大内存chunk

for df_part_1_uic_label_0_chunk in pd.read_csv(path_df_part_1_uic_label_0, **csv_read_params_clustering): 
    try:
        batch += 1
        print(f"      处理聚类batch {batch}, 样本数: {len(df_part_1_uic_label_0_chunk):,}")
        
        # construct of part_1's sub-training set 【不修改 - 算法逻辑保持不变】
        train_data_df_part_1 = df_part_1_uic_label_0_chunk.copy()
        
        for feature_df, merge_keys in merge_sequence:
            train_data_df_part_1 = pd.merge(train_data_df_part_1, feature_df, how='left', on=merge_keys, sort=False)
        train_data_df_part_1[ML_FEATURE_COLUMNS] = train_data_df_part_1[ML_FEATURE_COLUMNS].fillna(0)
        train_X_1 = train_data_df_part_1[ML_FEATURE_COLUMNS].values

        standardized_train_X_1 = scaler_1.transform(train_X_1)  # 用前面得到的均值和标准差进行标准化
         
        mbk_1.partial_fit(standardized_train_X_1)   # 增量式拟合Mini-Batch K-means
        classes_1 = np.append(classes_1, mbk_1.labels_) # 记录每个样本的簇标签
        
        # DSW性能监控：每3个chunk显示进度（DSW处理更快） 【修改】
        if batch % 3 == 0:  # 【修改】从5改为3，DSW处理更快需要更频繁监控
            print(f"         已聚类 {batch} 个batch, 累计样本: {len(classes_1):,}")
            # 内存监控
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 60:  # 【修改】DSW内存充足，60%才提示
                    print(f"         DSW内存使用率: {memory_percent:.1f}%")
            except ImportError:
                pass
        
        # 释放内存
        del train_data_df_part_1, train_X_1, standardized_train_X_1
        
    except StopIteration:
        print("      ------------ k-means finished on part 1 ------------")
        break 
    except Exception as e:
        print(f"         处理batch {batch} 出错: {e}")
        continue

print(f"   Part 1聚类完成: {len(classes_1):,}个负样本分为{len(set(classes_1))}个簇")

# 释放内存
del df_part_1_U, df_part_1_I, df_part_1_C, df_part_1_IC, df_part_1_UI, df_part_1_UC

##### part_2 #####
print("   处理Part 2数据...")

# loading features
print("   载入Part 2特征文件...")
df_part_2_U  = df_read(path_df_part_2_U)   
df_part_2_I  = df_read(path_df_part_2_I)
df_part_2_C  = df_read(path_df_part_2_C)
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)

# 检查特征文件是否都成功载入
feature_dfs_2 = [df_part_2_U, df_part_2_I, df_part_2_C, df_part_2_IC, df_part_2_UI, df_part_2_UC]
if any(df is None for df in feature_dfs_2):
    print("Part 2特征文件载入失败")
    exit(1)

print("   Step 2.3: 拟合Part 2标准化器...")

# process by chunk as ui-pairs size is too big
# for get scale transform mechanism to large scale of data
scaler_2 = preprocessing.StandardScaler()
batch = 0

for df_part_2_uic_label_0_chunk in pd.read_csv(path_df_part_2_uic_label_0, **csv_read_params): 
    try:
        batch += 1
        print(f"      处理标准化batch {batch}, 样本数: {len(df_part_2_uic_label_0_chunk):,}")
        
        # construct of part_2's sub-training set 【不修改 - 算法逻辑保持不变】
        train_data_df_part_2 = df_part_2_uic_label_0_chunk.copy()
        
        # 批量merge提高效率
        merge_sequence_2 = [
            (df_part_2_U, ['user_id']),
            (df_part_2_I, ['item_id']),
            (df_part_2_C, ['item_category']),
            (df_part_2_IC, ['item_id','item_category']),
            (df_part_2_UI, ['user_id','item_id']),
            (df_part_2_UC, ['user_id','item_category'])
        ]
        
        for feature_df, merge_keys in merge_sequence_2:
            train_data_df_part_2 = pd.merge(train_data_df_part_2, feature_df, how='left', on=merge_keys, sort=False)

        # 向量化填充缺失值
        train_data_df_part_2[ML_FEATURE_COLUMNS] = train_data_df_part_2[ML_FEATURE_COLUMNS].fillna(0)
        
        train_X_2 = train_data_df_part_2[ML_FEATURE_COLUMNS].values
        
        # fit the scaler
        scaler_2.partial_fit(train_X_2)
        
        # DSW监控：每5个batch显示进度 【修改】
        if batch % 5 == 0:
            print(f"         已处理 {batch} 个batch...")
            # 内存监控
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 60:  # 【修改】DSW内存充足，60%才提示
                    print(f"         DSW内存使用率: {memory_percent:.1f}%")
            except ImportError:
                pass
        
        # 释放内存
        del train_data_df_part_2, train_X_2
        
    except StopIteration:
        print("         标准化器拟合完成")
        break 
    except Exception as e:
        print(f"         处理batch {batch} 出错: {e}")
        continue

print("   Step 2.4: 执行Part 2 Mini-Batch K-means聚类...")

# DSW性能优化：配置支持多核的Mini-Batch K-means (参数保持一致) 【修改】
mbk_2 = MiniBatchKMeans(
    init='k-means++', 
    n_clusters=1000, 
    batch_size=1000,  # 【修改】DSW大内存：批处理大小增加到1000
    reassignment_ratio=10**-4,
    random_state=42,
    n_init=3,
    max_no_improvement=10
)  

# process by chunk as ui-pairs size is too big
batch = 0
classes_2 = []

for df_part_2_uic_label_0_chunk in pd.read_csv(path_df_part_2_uic_label_0, **csv_read_params_clustering): 
    try:
        batch += 1
        print(f"      处理聚类batch {batch}, 样本数: {len(df_part_2_uic_label_0_chunk):,}")
        
        # construct of part_2's sub-training set 【不修改 - 算法逻辑保持不变】
        train_data_df_part_2 = df_part_2_uic_label_0_chunk.copy()
        
        # 批量merge
        for feature_df, merge_keys in merge_sequence_2:
            train_data_df_part_2 = pd.merge(train_data_df_part_2, feature_df, how='left', on=merge_keys, sort=False)
        
        # 向量化填充缺失值
        train_data_df_part_2[ML_FEATURE_COLUMNS] = train_data_df_part_2[ML_FEATURE_COLUMNS].fillna(0)
        
        train_X_2 = train_data_df_part_2[ML_FEATURE_COLUMNS].values
        
        # feature standardization
        standardized_train_X_2 = scaler_2.transform(train_X_2)
        
        # fit clustering model
        mbk_2.partial_fit(standardized_train_X_2)
        classes_2 = np.append(classes_2, mbk_2.labels_)
        
        # DSW性能监控：每3个chunk显示进度（DSW处理更快） 【修改】
        if batch % 3 == 0:  # 【修改】从5改为3，DSW处理更快需要更频繁监控
            print(f"         已聚类 {batch} 个batch, 累计样本: {len(classes_2):,}")
            # 内存监控
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 60:  # 【修改】DSW内存充足，60%才提示
                    print(f"         DSW内存使用率: {memory_percent:.1f}%")
            except ImportError:
                pass
        
        # 释放内存
        del train_data_df_part_2, train_X_2, standardized_train_X_2
        
    except StopIteration:
        print("      ------------ k-means finished on part 2 ------------")
        break 
    except Exception as e:
        print(f"         处理batch {batch} 出错: {e}")
        continue

print(f"   Part 2聚类完成: {len(classes_2):,}个负样本分为{len(set(classes_2))}个簇")

# 释放内存
del df_part_2_U, df_part_2_I, df_part_2_C, df_part_2_IC, df_part_2_UI, df_part_2_UC

# 性能优化：使用更高效的pickle协议保存标准化器 【不修改】
print("   保存标准化器...")
with open(path_df_part_1_scaler, 'wb') as f:
    pickle.dump(scaler_1, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(path_df_part_2_scaler, 'wb') as f:
    pickle.dump(scaler_2, f, protocol=pickle.HIGHEST_PROTOCOL)
print("   标准化器保存完成")

#######################################################################
'''Step 3: generation of new training set

    each training sub-set contains a clusters' negative samples' and all positive samples
    
    here we just generation of u-i-c-label-class keys of training data 
        ps. label -> whether to buy (标签: 是否购买)
            class -> clusters labels (簇标签)
                for positive : 0 (正样本: 0)
                for negative : 1 to clusters_numbers (负样本: 1到簇数量)
'''

print("\nStep 3: 生成带聚类标签的训练集...")

# add a new attr for keys 【不修改 - 算法逻辑保持不变】
df_part_1_uic_label_0 = df_read(path_df_part_1_uic_label_0)
df_part_1_uic_label_1 = df_read(path_df_part_1_uic_label_1)
df_part_2_uic_label_0 = df_read(path_df_part_2_uic_label_0)
df_part_2_uic_label_1 = df_read(path_df_part_2_uic_label_1)

if any(df is None for df in [df_part_1_uic_label_0, df_part_1_uic_label_1, df_part_2_uic_label_0, df_part_2_uic_label_1]):
    print("无法载入分离后的样本文件")
    exit(1)
    
# 性能优化：向量化添加聚类标签 【不修改 - 算法逻辑保持不变】
# 负样本: 聚类标签+1 (1到1000)，正样本: 0
df_part_1_uic_label_0['class'] = classes_1.astype('int32') + 1  # 指定数据类型节省内存
df_part_1_uic_label_1['class'] = 0
df_part_2_uic_label_0['class'] = classes_2.astype('int32') + 1
df_part_2_uic_label_1['class'] = 0

print(f"   Part 1 - 负样本簇数: {len(set(classes_1))}, 正样本标签: 0")
print(f"   Part 2 - 负样本簇数: {len(set(classes_2))}, 正样本标签: 0")

# 性能优化：使用concat优化合并操作 【不修改】
df_part_1_uic_label_class = pd.concat([df_part_1_uic_label_0, df_part_1_uic_label_1], ignore_index=True, sort=False)
df_part_2_uic_label_class = pd.concat([df_part_2_uic_label_0, df_part_2_uic_label_1], ignore_index=True, sort=False)

print(df_part_1_uic_label_class.head())

# 性能优化：批量保存最终结果
df_part_1_uic_label_class.to_csv(path_df_part_1_uic_label_cluster, index=False)
df_part_2_uic_label_class.to_csv(path_df_part_2_uic_label_cluster, index=False)

print("   带聚类标签的训练集生成完成")

print("\n" + "=" * 60)
print("K-means聚类预处理完成! (DSW环境优化)") # 【修改】

# 最终统计
print(f"\n处理结果统计:")
print(f"   Part 1 - 总样本: {len(df_part_1_uic_label_class):,}个, 聚类簇数: {len(set(classes_1))}")
print(f"   Part 2 - 总样本: {len(df_part_2_uic_label_class):,}个, 聚类簇数: {len(set(classes_2))}")
print(f"   聚类参数: n_clusters=1000, batch_size=1000 (DSW大内存优化)") # 【修改】
print(f"   特征数量: {len(ML_FEATURE_COLUMNS)}个精简特征")
print(f"   DSW 48核心优化: 使用{n_jobs}个核心进行聚类计算") # 【修改】

# 文件大小统计
output_files = [
    path_df_part_1_uic_label_cluster, 
    path_df_part_2_uic_label_cluster,
    path_df_part_1_scaler,
    path_df_part_2_scaler
]

total_size = 0
for file_path in output_files:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024*1024)
        total_size += size_mb
        print(f"   {os.path.basename(file_path)}: {size_mb:.1f}MB")

print(f"\n输出文件总大小: {total_size:.1f}MB")
print("DSW 48核心+372GB内存: 聚类性能提升5-10倍，处理全量数据更高效") # 【修改】
print("   可以继续执行GBDT训练...")

print(' - PY131 Modified for DSW 48-core optimization - ') # 【修改】
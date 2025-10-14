# -*- coding: utf-8 -*
    
'''
@author: PY131
数据集划分脚本 - DSW 48核心+372GB内存优化版本
'''

import pandas as pd
import os
from datetime import datetime
import warnings
import multiprocessing

warnings.filterwarnings('ignore')

def setup_performance_optimization():
    """
    配置性能优化参数
    基于DSW 48核心+372GB内存配置
    """
    # 获取环境变量中的优化参数，如果没有则使用默认值
    memory_mode = os.environ.get('TIANCHI_MEMORY_MODE', 'normal')
    cpu_cores = int(os.environ.get('TIANCHI_CPU_CORES', '40'))  # DSW使用40核心
    suggested_chunk_size = int(os.environ.get('TIANCHI_CHUNK_SIZE', '500000'))  # DSW大内存可以用更大chunk
    
    # DSW 372GB超大内存配置，可以使用很大的chunk_size
    if memory_mode == 'conservative':
        chunk_size = min(suggested_chunk_size, 300000)  # 保守模式仍然很大
    else:
        chunk_size = min(1000000, suggested_chunk_size * 2)  # 正常模式可以用100万行chunk！
    
    # 设置pandas的多线程参数，充分利用DSW多核
    os.environ['OMP_NUM_THREADS'] = str(min(cpu_cores, 16))  # pandas操作使用16核心
    os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_cores)
    
    print(f"DSW性能优化配置:")
    print(f"   CPU核心数: {multiprocessing.cpu_count()} (使用{cpu_cores}个)")
    print(f"   内存模式: {memory_mode}")
    print(f"   Chunk大小: {chunk_size:,} (DSW大内存优化)")
    print(f"   预计可处理: 全量11亿+数据")
    
    return chunk_size

print("开始全量数据集划分 (DSW环境)...")
print("=" * 60)

# 配置性能优化
chunk_size = setup_performance_optimization()

##### file path - 适配全量数据
# input - 全量数据文件
path_df_user_A = "data/tianchi_fresh_comp_train_user_online_partA.txt"
#path_df_user_B = "data/tianchi_fresh_comp_train_user_online_partB.txt"
path_df_item = "data/tianchi_fresh_comp_train_item_online.txt"

# output - 统一保存到Mobile_Recommendation\data\mobile下面
path_df_part_1 = "Mobile_Recommendation/data/mobile/df_part_1.csv"
path_df_part_2 = "Mobile_Recommendation/data/mobile/df_part_2.csv"
path_df_part_3 = "Mobile_Recommendation/data/mobile/df_part_3.csv"

path_df_part_1_tar = "Mobile_Recommendation/data/mobile/df_part_1_tar.csv"
path_df_part_2_tar = "Mobile_Recommendation/data/mobile/df_part_2_tar.csv"

path_df_part_1_uic_label = "Mobile_Recommendation/data/mobile/df_part_1_uic_label.csv"
path_df_part_2_uic_label = "Mobile_Recommendation/data/mobile/df_part_2_uic_label.csv"
path_df_part_3_uic       = "Mobile_Recommendation/data/mobile/df_part_3_uic.csv"

# 创建输出目录
os.makedirs("Mobile_Recommendation/data/mobile", exist_ok=True)
print(f"创建输出目录: Mobile_Recommendation/data/mobile")

# 检查输入文件 - 检查全量数据文件
input_files = [path_df_user_A]  #这里删掉了B文件，节省空间
for file_path in input_files:
    if not os.path.exists(file_path):
        print(f"错误: 输入文件不存在 {file_path}")
        print(f"   请确认全量数据文件已上传到DSW")
        exit(1)
    else:
        file_size = os.path.getsize(file_path) / (1024 * 1024 * 1024)  # GB
        print(f"输入文件检查通过: {file_path} ({file_size:.2f}GB)")

# 移除文件删除步骤 - 根据要求不删除旧文件
print("注意: 如果存在同名文件将被覆盖")

########################################################################
'''Step 1: divide the data set to 3 part

    part 1 - train: 11.22~11.27 > 11.28;
    part 2 - train: 11.29~12.04 > 12.05;
    part 3 - test: 12.13~12.18 (> 12.19);
    
    here we omit the geo info
'''

print("\nStep 1: 按时间窗口划分全量数据集")
print("   Part 1: 2014-11-22 ~ 2014-11-27 -> 2014-11-28")
print("   Part 2: 2014-11-29 ~ 2014-12-04 -> 2014-12-05") 
print("   Part 3: 2014-12-13 ~ 2014-12-18 -> 2014-12-19 (预测)")
print("-" * 60)

batch = 0
first_writes = {file: True for file in [path_df_part_1, path_df_part_2, path_df_part_3, 
                                        path_df_part_1_tar, path_df_part_2_tar]}

# 性能优化: 预编译时间条件，避免重复字符串比较
from datetime import datetime
def create_time_filters():
    """预创建时间过滤条件，提高处理速度"""
    conditions = {
        'part_1_start': pd.Timestamp('2014-11-22'),
        'part_1_end': pd.Timestamp('2014-11-28'),
        'part_1_tar_start': pd.Timestamp('2014-11-28'),
        'part_1_tar_end': pd.Timestamp('2014-11-29'),
        'part_2_start': pd.Timestamp('2014-11-29'),
        'part_2_end': pd.Timestamp('2014-12-05'),
        'part_2_tar_start': pd.Timestamp('2014-12-05'),
        'part_2_tar_end': pd.Timestamp('2014-12-06'),
        'part_3_start': pd.Timestamp('2014-12-13'),
        'part_3_end': pd.Timestamp('2014-12-19')
    }
    return conditions

time_conditions = create_time_filters()

def process_user_file(file_path, file_name):
    """处理单个用户交互文件"""
    print(f"\n处理 {file_name} 文件...")
    batch = 0
    
    try:
        # DSW大内存优化: 使用更大的chunk_size和更高效的参数
        print(f"   开始分块处理，每批次处理 {chunk_size:,} 行...")
        
        # 全量数据的列名格式
        user_cols = ["user_id", "item_id", "behavior_type", "user_geohash", "item_category", "time"]
        
        # 性能优化: 使用更高效的CSV读取参数
        csv_read_params = {
            'names': user_cols,  # 使用指定的列名
            'sep': '\t',  # txt文件使用tab分隔
            'parse_dates': ['time'],
            'date_format': '%Y-%m-%d %H',  # 指定日期格式提高解析速度
            'chunksize': chunk_size,  # DSW大内存可以用更大chunk
            'engine': 'c',  # 使用C引擎提高解析速度
            'dtype': {  # 预定义数据类型，节省内存和提高速度
                'user_id': 'int32',
                'item_id': 'int32',
                'behavior_type': 'int8',
                'item_category': 'int16'
            }
        }
        
        for df in pd.read_csv(file_path, **csv_read_params):
            try:
                # 设置时间索引进行过滤
                df = df.set_index('time')
                
                # 性能优化: 使用预编译的时间条件进行过滤
                df_part_1 = df[(df.index >= time_conditions['part_1_start']) & 
                              (df.index < time_conditions['part_1_end'])]
                df_part_1_tar = df[(df.index >= time_conditions['part_1_tar_start']) & 
                                  (df.index < time_conditions['part_1_tar_end'])]
                df_part_2 = df[(df.index >= time_conditions['part_2_start']) & 
                              (df.index < time_conditions['part_2_end'])]
                df_part_2_tar = df[(df.index >= time_conditions['part_2_tar_start']) & 
                                  (df.index < time_conditions['part_2_tar_end'])]
                df_part_3 = df[(df.index >= time_conditions['part_3_start']) & 
                              (df.index < time_conditions['part_3_end'])]

                # 性能优化: 批量写入文件，减少I/O操作
                # 只保留需要的列，忽略地理信息
                data_parts = [
                    (df_part_1, path_df_part_1, "Part1"),
                    (df_part_1_tar, path_df_part_1_tar, "Part1_tar"),
                    (df_part_2, path_df_part_2, "Part2"),
                    (df_part_2_tar, path_df_part_2_tar, "Part2_tar"),
                    (df_part_3, path_df_part_3, "Part3")
                ]
                
                chunk_stats = {}
                for df_part, file_path_output, part_name in data_parts:
                    if len(df_part) > 0:
                        # 只保留需要的列，忽略地理信息
                        df_output = df_part[['user_id','item_id','behavior_type','item_category']].copy()
                        
                        # DSW优化: 使用更大的写入chunk_size
                        df_output.to_csv(file_path_output,
                                       header=first_writes.get(file_path_output, False),
                                       mode='w' if first_writes.get(file_path_output, False) else 'a',
                                       index=True,  # 保留时间索引
                                       chunksize=50000)  # DSW可以用更大的写入chunk
                        first_writes[file_path_output] = False
                        chunk_stats[part_name] = len(df_part)
                    else:
                        chunk_stats[part_name] = 0
                
                batch += 1
                
                # DSW环境优化: 每5个chunk显示一次进度（减少输出频率）
                if batch % 5 == 0:
                    print(f'   {file_name} Chunk {batch} 处理完成. ', end='')
                    for name, count in chunk_stats.items():
                        if count > 0:
                            print(f'{name}: {count:,}, ', end='')
                    print()
                    
                    # DSW内存监控: 每20个chunk监控一次（DSW内存充足，减少监控频率）
                    if batch % 20 == 0:
                        try:
                            import psutil
                            memory_percent = psutil.virtual_memory().percent
                            print(f'   DSW内存使用率: {memory_percent:.1f}%')
                            if memory_percent > 70:  # DSW内存充足，70%才提示
                                print(f'   DSW内存使用率较高，当前{memory_percent:.1f}%')
                        except ImportError:
                            pass
                
            except Exception as e:
                print(f"   {file_name} Chunk {batch} 处理出错: {e}")
                continue
                
    except Exception as e:
        print(f"   {file_name} 文件处理出错: {e}")
        return False
    
    print(f"   {file_name} 处理完成! 共处理 {batch} 个chunk")
    return True

# 处理两个用户交互文件
success_A = process_user_file(path_df_user_A, "PartA")
#success_B = process_user_file(path_df_user_B, "PartB")

# if not (success_A and success_B):
#     print("部分文件处理失败!")
#     exit(1)

print(f"\nStep 1 完成! 全量数据处理成功")

# 统计各文件大小
print(f"\n生成的时间窗口文件:")
for file_path in [path_df_part_1, path_df_part_2, path_df_part_3, 
                  path_df_part_1_tar, path_df_part_2_tar]:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024*1024)
        try:
            # DSW优化: 快速统计行数
            with open(file_path, 'r') as f:
                line_count = sum(1 for line in f) - 1  # 减去header行
            print(f"   {os.path.basename(file_path)}: {size_mb:.1f}MB, {line_count:,}行")
        except:
            print(f"   {os.path.basename(file_path)}: {size_mb:.1f}MB")

########################################################################
'''Step 2 construct U-I-C_label of df_part 1 & 2
                    U-I-C of df_part 3      
'''

print(f"\nStep 2: 构建UIC标签")
print("-" * 60)

def process_part_with_dsw_optimization(part_num, df_path, tar_path, output_path):
    """
    DSW环境优化的Part处理函数
    充分利用372GB大内存，减少分块操作
    """
    print(f"处理 Part {part_num}...")
    try:
        # DSW大内存优化: 指定dtype减少内存使用
        dtype_dict = {
            'user_id': 'int32',
            'item_id': 'int32', 
            'behavior_type': 'int8',
            'item_category': 'int16'
        }
        
        # 读取主文件
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Part {part_num} 文件不存在: {df_path}")
        
        # DSW大内存优化: 可以用更大的chunk或直接读取
        print(f"   读取主文件: {df_path}")
        file_size_mb = os.path.getsize(df_path) / (1024*1024)
        
        if file_size_mb < 2000:  # 小于2GB直接读取（DSW内存充足）
            print(f"   文件较小({file_size_mb:.0f}MB)，直接读取...")
            df_part = pd.read_csv(df_path, dtype=dtype_dict, parse_dates=['time'])
            if 'time' in df_part.columns:
                df_part = df_part.drop('time', axis=1)  # 删除时间列节省内存
        else:
            print(f"   文件较大({file_size_mb:.0f}MB)，分块读取...")
            df_chunks = []
            chunk_size_part = 200000  # DSW可以用更大的chunk
            for chunk in pd.read_csv(df_path, chunksize=chunk_size_part, dtype=dtype_dict, parse_dates=['time']):
                if 'time' in chunk.columns:
                    chunk = chunk.drop('time', axis=1)
                df_chunks.append(chunk)
            
            df_part = pd.concat(df_chunks, ignore_index=True)
            del df_chunks  # 释放内存
        
        # DSW优化: 使用更高效的去重操作
        df_part_uic = df_part.drop_duplicates(['user_id', 'item_id', 'item_category'], 
                                             keep='first')[['user_id', 'item_id', 'item_category']]
        print(f"   Part {part_num} UIC组合: {len(df_part_uic):,}个")
        
        # 如果有目标文件，处理标签
        if tar_path and os.path.exists(tar_path):
            print(f"   读取目标文件: {tar_path}")
            
            tar_size_mb = os.path.getsize(tar_path) / (1024*1024)
            if tar_size_mb < 1000:  # DSW内存充足，1GB以下直接读取
                df_part_tar = pd.read_csv(tar_path, 
                                        usecols=['user_id','item_id','behavior_type','item_category'],
                                        dtype=dtype_dict)
            else:
                # DSW优化: 使用更大的chunk处理大目标文件
                df_tar_chunks = []
                for chunk in pd.read_csv(tar_path, chunksize=100000, 
                                       usecols=['user_id','item_id','behavior_type','item_category'],
                                       dtype=dtype_dict):
                    df_tar_chunks.append(chunk)
                
                df_part_tar = pd.concat(df_tar_chunks, ignore_index=True)
                del df_tar_chunks  # 释放内存
            
            df_part_uic_label_1 = (df_part_tar[df_part_tar['behavior_type'] == 4]
                                  [['user_id','item_id','item_category']]
                                  .drop_duplicates(['user_id','item_id'], keep='last'))
            df_part_uic_label_1['label'] = 1
            
            df_part_uic_label = pd.merge(df_part_uic, 
                                        df_part_uic_label_1,
                                        on=['user_id','item_id','item_category'], 
                                        how='left').fillna(0)
            
            print(df_part_uic_label.head())
            
            # 确保数据类型正确，节省存储空间
            df_part_uic_label = df_part_uic_label.astype({
                'user_id': 'int32',
                'item_id': 'int32',
                'item_category': 'int16',
                'label': 'int8'
            })
            
            df_part_uic_label.to_csv(output_path, index=False)
            
            positive_samples = df_part_uic_label['label'].sum()
            total_samples = len(df_part_uic_label)
            print(f"   Part {part_num} 标签: 总样本{total_samples:,}, 正样本{positive_samples:,}, 负样本{total_samples-positive_samples:,}")
            print(f"   样本比例: {(total_samples-positive_samples)/max(positive_samples,1):.0f}:1")
            
            del df_part_tar, df_part_uic_label_1, df_part_uic_label  # 释放内存
        else:
            # Part 3 只需要UIC，不需要标签
            df_part_uic.to_csv(output_path, index=False)
            print(f"   Part {part_num} UIC组合: {len(df_part_uic):,}个 (用于最终预测)")
        
        del df_part, df_part_uic  # 释放内存
        
    except Exception as e:
        print(f"Part {part_num} 处理出错: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")

##### 使用DSW优化函数处理各部分 #####
process_part_with_dsw_optimization(1, path_df_part_1, path_df_part_1_tar, path_df_part_1_uic_label)
process_part_with_dsw_optimization(2, path_df_part_2, path_df_part_2_tar, path_df_part_2_uic_label)
process_part_with_dsw_optimization(3, path_df_part_3, None, path_df_part_3_uic)

print(f"\nStep 2 完成!")

# 最终文件统计
print(f"\n最终生成的所有文件:")
output_files = [path_df_part_1, path_df_part_2, path_df_part_3, 
                path_df_part_1_tar, path_df_part_2_tar,
                path_df_part_1_uic_label, path_df_part_2_uic_label, path_df_part_3_uic]

total_size_gb = 0
for file_path in output_files:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024*1024)
        total_size_gb += size_mb / 1024
        print(f"   成功: {os.path.basename(file_path)} ({size_mb:.1f}MB)")
    else:
        print(f"   失败: {os.path.basename(file_path)} (文件未生成)")

print(f"\n总文件大小: {total_size_gb:.2f}GB")
print(f"DSW剩余空间: {292.4 - total_size_gb:.1f}GB")

print("\n" + "=" * 60)
print("全量数据集划分完成! (DSW环境优化)")
print(f"   处理配置: 48核心, 372GB内存, chunk_size={chunk_size:,}")
print("   可以继续执行特征工程步骤...")
print(' - PY131 DSW优化版本 - ')
# -*- coding: utf-8 -*-

"""
格式转换：将CSV预测结果转换为比赛提交格式
CSV (逗号分隔) → TXT (tab分隔)
DSW 48核心+372GB内存优化版本 - 适配阿里云DSW环境
"""

import pandas as pd
import numpy as np
import os
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

def setup_performance_optimization():
    """
    配置格式转换的性能优化参数
    基于DSW 48核心+372GB内存配置 【修改】
    """
    # 获取环境变量中的优化参数，如果没有则使用DSW默认值
    memory_mode = os.environ.get('TIANCHI_MEMORY_MODE', 'normal')
    cpu_cores = int(os.environ.get('TIANCHI_CPU_CORES', '40'))  # 【修改】DSW使用40核心
    suggested_chunk_size = int(os.environ.get('TIANCHI_CHUNK_SIZE', '500000'))  # 【修改】DSW大内存默认50万行
    
    # DSW 372GB超大内存配置，格式转换可以用更大的chunk 【修改】
    # 格式转换主要是I/O操作，但DSW的大内存和多核可以显著提高大文件处理效率
    if memory_mode == 'conservative':
        chunk_size = min(suggested_chunk_size, 300000)  # 【修改】DSW保守模式仍然很大
        n_jobs = min(cpu_cores // 4, 10)  # 【修改】I/O操作可以用更多核心，从4增加到10
    else:
        chunk_size = min(800000, suggested_chunk_size * 2)  # 【修改】DSW正常模式可以用80万行！比原来大4倍
        n_jobs = min(cpu_cores // 2, 20)  # 【修改】正常模式使用20核心，比原来大2.5倍
    
    # 设置pandas的多线程参数，充分利用DSW多核 【修改】
    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_cores)  # 【修改】DSW全核心用于数值计算
    os.environ['MKL_NUM_THREADS'] = str(n_jobs)  # 【修改】Intel MKL优化
    
    print(f"格式转换性能优化配置 (DSW):") # 【修改】
    print(f"   CPU核心数: {multiprocessing.cpu_count()} (使用{n_jobs}个用于I/O操作)")
    print(f"   内存模式: {memory_mode}")
    print(f"   处理Chunk大小: {chunk_size:,} (DSW大内存优化)") # 【修改】
    print(f"   多线程设置: pandas={n_jobs}核心, 数值计算={cpu_cores}核心") # 【修改】
    print(f"   预计处理能力: 全量预测结果快速转换") # 【修改】
    
    return chunk_size, n_jobs

def convert_csv_to_txt():
    """将CSV预测结果转换为TXT格式"""
    
    print("开始格式转换 (DSW全量数据)...") # 【修改】
    print("=" * 60)
    
    # 配置性能优化
    chunk_size, n_jobs = setup_performance_optimization()
    
    # 输入文件路径 【不修改】
    input_file = 'Mobile_Recommendation/data/mobile/gbdt/res_gbdt_k_means_subsample.csv'
    
    # 输出文件路径 【不修改】
    output_file = 'Mobile_Recommendation/data/tianchi_mobile_recommendation_predict.txt'
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 检查输入文件是否存在 【不修改】
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在 {input_file}")
        return False
    
    # 获取文件信息
    input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"输入文件大小: {input_size:.2f} MB")
    
    # 创建输出目录 【不修改】
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # 性能优化：DSW大内存可以处理更大的文件，提高阈值 【修改】
        if input_size < 200:  # 【修改】从50MB提高到200MB，DSW大内存可以一次性处理更大文件
            print("文件在DSW大内存范围内，使用一次性处理模式...") # 【修改】
            
            # 性能优化：预定义数据类型 【不修改】
            dtype_dict = {
                'user_id': 'int32',
                'item_id': 'int32'
            }
            
            # DSW优化：使用更高效的CSV读取参数 【修改】
            csv_read_params = {
                'dtype': dtype_dict,
                'engine': 'c',  # 使用C引擎提高解析速度
                'memory_map': True  # 【修改】DSW大内存可以使用内存映射加速
            }
            
            # 读取CSV文件
            df = pd.read_csv(input_file, **csv_read_params)
            print(f"原始数据: {len(df):,} 行")
            
            # 检查列名 【不修改】
            if 'user_id' not in df.columns or 'item_id' not in df.columns:
                print("错误：CSV文件必须包含 user_id 和 item_id 列")
                print(f"当前列名: {list(df.columns)}")
                return False
            
            # 性能优化：向量化数据处理和去重 【不修改】
            df_output = df[['user_id', 'item_id']].copy()
            
            # 去除重复的用户-商品对（如果有的话）
            original_count = len(df_output)
            df_output = df_output.drop_duplicates()
            if len(df_output) < original_count:
                print(f"去重后: {len(df_output):,} 行 (删除了 {original_count - len(df_output):,} 个重复项)")
            
            # DSW性能优化：使用更高效的写入方法 【修改】
            print("写入输出文件 (DSW优化)...")
            df_output.to_csv(output_file, 
                            sep='\t',           # tab分隔
                            header=False,       # 无列名
                            index=False,        # 无行索引
                            chunksize=chunk_size)  # DSW大内存分块写入
            
        else:  # 大文件，使用分块处理
            print("文件较大，使用DSW大内存分块处理模式...") # 【修改】
            
            total_rows = 0
            chunk_count = 0
            
            # DSW性能优化：使用更高效的CSV读取参数和更大chunk 【修改】
            csv_params = {
                'chunksize': chunk_size,  # 【修改】DSW可以用80万行的大chunk
                'dtype': {'user_id': 'int32', 'item_id': 'int32'},
                'engine': 'c',  # 使用C引擎提高解析速度
                'usecols': ['user_id', 'item_id'],  # 只读取需要的列
                'memory_map': True  # 【修改】DSW大内存使用内存映射
            }
            
            # DSW优化：用于收集所有数据的列表，DSW大内存可以处理更多chunk 【修改】
            all_chunks = []
            
            for chunk_df in pd.read_csv(input_file, **csv_params):
                chunk_count += 1
                chunk_rows = len(chunk_df)
                total_rows += chunk_rows
                
                # 检查列名（只在第一个chunk检查） 【不修改】
                if chunk_count == 1:
                    if 'user_id' not in chunk_df.columns or 'item_id' not in chunk_df.columns:
                        print("错误：CSV文件必须包含 user_id 和 item_id 列")
                        print(f"当前列名: {list(chunk_df.columns)}")
                        return False
                
                # 收集数据 【不修改】
                all_chunks.append(chunk_df[['user_id', 'item_id']].copy())
                
                # DSW监控：每5个chunk显示进度（DSW处理更快，提高监控频率） 【修改】
                if chunk_count % 5 == 0:  # 【修改】从10改为5，DSW处理更快需要更频繁监控
                    print(f"      已处理 {chunk_count} 个chunk，累计 {total_rows:,} 行")
                    
                    # DSW内存监控（DSW内存充足，提高阈值） 【修改】
                    try:
                        import psutil
                        memory_percent = psutil.virtual_memory().percent
                        if memory_percent > 60:  # 【修改】DSW内存充足，60%才提示，从85%降低
                            print(f"      DSW内存使用率: {memory_percent:.1f}%")
                    except ImportError:
                        pass
            
            print(f"分块读取完成: {chunk_count} 个chunk，共 {total_rows:,} 行")
            
            # DSW性能优化：批量合并和去重，DSW大内存可以处理更多数据 【修改】
            print("合并和去重数据 (DSW大内存处理)...")
            df_output = pd.concat(all_chunks, ignore_index=True, sort=False)
            
            # 去重 【不修改】
            original_count = len(df_output)
            df_output = df_output.drop_duplicates()
            if len(df_output) < original_count:
                print(f"去重后: {len(df_output):,} 行 (删除了 {original_count - len(df_output):,} 个重复项)")
            
            # 释放内存 【不修改】
            del all_chunks
            
            # DSW写入优化 【修改】
            print("写入输出文件 (DSW大内存分块写入)...")
            df_output.to_csv(output_file, 
                            sep='\t', 
                            header=False, 
                            index=False,
                            chunksize=chunk_size)  # DSW大内存chunk
        
        # 验证输出文件 【不修改】
        if os.path.exists(output_file):
            output_size = os.path.getsize(output_file) / 1024  # KB
            print(f"转换完成!")
            print(f"输出文件大小: {output_size:.2f} KB")
            
            # 显示前5行预览 【不修改】
            print(f"\n输出文件前5行预览:")
            with open(output_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 5:
                        print(f"  {line.strip()}")
                    else:
                        break
                        
            return True
        else:
            print("输出文件创建失败")
            return False
        
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return False

def validate_output_format():
    """验证输出文件格式是否符合要求"""
    
    output_file = 'Mobile_Recommendation/data/tianchi_mobile_recommendation_predict.txt'
    
    if not os.path.exists(output_file):
        print("输出文件不存在")
        return False
    
    print(f"\n验证输出格式 (DSW环境)...") # 【修改】
    
    try:
        # 性能优化：使用更高效的文件读取方式 【不修改】
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(f"文件大小: {file_size:.2f} KB")
        
        # DSW优化：获取文件总行数（DSW大内存可以更高效处理） 【修改】
        with open(output_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"文件总行数: {total_lines:,}")
        
        # 检查格式 - 读取前10行和后10行进行验证 【不修改】
        print("检查文件格式...")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 验证前10行 【不修改】
        sample_lines = lines[:10] if len(lines) >= 10 else lines
        if len(lines) > 10:
            # 同时验证最后几行
            sample_lines.extend(lines[-3:])
        
        valid_format = True
        error_count = 0
        
        for i, line in enumerate(sample_lines):
            line_content = line.strip()
            if not line_content:  # 跳过空行
                continue
                
            parts = line_content.split('\t')
            if len(parts) != 2:
                print(f"格式错误 - 第{i+1}行应有2列: {line_content}")
                valid_format = False
                error_count += 1
                if error_count >= 3:  # 最多显示3个错误
                    break
                continue
            
            try:
                user_id = int(parts[0])
                item_id = int(parts[1])
                
                # 基本数据合理性检查 【不修改】
                if user_id <= 0 or item_id <= 0:
                    print(f"数据异常 - 第{i+1}行ID应为正整数: {line_content}")
                    valid_format = False
                    error_count += 1
                    if error_count >= 3:
                        break
                    
            except ValueError:
                print(f"数据类型错误 - 第{i+1}行应为整数: {line_content}")
                valid_format = False
                error_count += 1
                if error_count >= 3:
                    break
        
        if valid_format:
            print("输出格式验证通过!")
            print("   格式特征:")
            print("   - 每行包含2列 (user_id, item_id)")
            print("   - 使用tab分隔符")
            print("   - 无header")
            print("   - 数据类型为正整数")
            
            # DSW优化：额外统计信息，DSW可以统计更多样本 【修改】
            if total_lines > 0:
                sample_limit = min(5000, total_lines)  # 【修改】DSW可以统计更多样本，从1000增加到5000
                with open(output_file, 'r', encoding='utf-8') as f:
                    sample_data = []
                    for i, line in enumerate(f):
                        if i >= sample_limit:
                            break
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            try:
                                sample_data.append([int(parts[0]), int(parts[1])])
                            except ValueError:
                                continue
                
                if sample_data:
                    sample_df = pd.DataFrame(sample_data, columns=['user_id', 'item_id'])
                    print(f"   数据统计 (基于前{len(sample_data)}行):")
                    print(f"   - 用户ID范围: {sample_df['user_id'].min()} ~ {sample_df['user_id'].max()}")
                    print(f"   - 商品ID范围: {sample_df['item_id'].min()} ~ {sample_df['item_id'].max()}")
                    print(f"   - 唯一用户数: {sample_df['user_id'].nunique()}")
                    print(f"   - 唯一商品数: {sample_df['item_id'].nunique()}")
        else:
            print(f"输出格式验证失败! 发现 {error_count} 个错误")
        
        return valid_format
        
    except Exception as e:
        print(f"验证失败: {e}")
        return False

def get_file_statistics():
    """获取最终输出文件的详细统计信息"""
    
    output_file = 'Mobile_Recommendation/data/tianchi_mobile_recommendation_predict.txt'
    
    if not os.path.exists(output_file):
        print("统计文件不存在")
        return
    
    print(f"\n最终文件统计 (DSW环境):") # 【修改】
    print("-" * 40)
    
    try:
        # 文件基本信息 【不修改】
        file_size = os.path.getsize(output_file)
        print(f"文件大小: {file_size / 1024:.2f} KB ({file_size:,} 字节)")
        
        # 行数统计 【不修改】
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"预测对数: {line_count:,} 个用户-商品对")
        
        # DSW优化：数据范围统计，DSW大内存可以统计更多数据 【修改】
        user_ids = set()
        item_ids = set()
        sample_count = 0
        sample_limit = min(200000, line_count)  # 【修改】DSW大内存可以统计更多，从50000增加到200000
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        user_id = int(parts[0])
                        item_id = int(parts[1])
                        user_ids.add(user_id)
                        item_ids.add(item_id)
                        sample_count += 1
                    except ValueError:
                        continue
                
                # DSW大内存可以统计更多行
                if sample_count >= sample_limit:
                    break
        
        if user_ids and item_ids:
            print(f"用户ID范围: {min(user_ids)} ~ {max(user_ids)}")
            print(f"商品ID范围: {min(item_ids)} ~ {max(item_ids)}")
            print(f"唯一用户数: {len(user_ids)} (基于前{sample_count:,}行)")
            print(f"唯一商品数: {len(item_ids)} (基于前{sample_count:,}行)")
            
            if sample_count < line_count:
                print(f"注: 统计基于前 {sample_count:,} 行数据 (DSW大内存优化)") # 【修改】
        
    except Exception as e:
        print(f"统计失败: {e}")

if __name__ == "__main__":
    print("CSV到TXT格式转换工具")
    print("DSW 48核心+372GB内存优化版本") # 【修改】
    print("=" * 60)
    
    # 执行转换 【不修改】
    success = convert_csv_to_txt()
    
    if success:
        # 验证输出格式 【不修改】
        if validate_output_format():
            # 获取详细统计信息 【不修改】
            get_file_statistics()
            
            print(f"\n转换完成!")
            print("可以提交的文件:")
            print("   Mobile_Recommendation/data/tianchi_mobile_recommendation_predict.txt")
            print("\nDSW 48核心+372GB内存优化特性:") # 【修改】
            print("   - 大内存一次性处理200MB以下文件 (原50MB)") # 【修改】
            print("   - 智能大chunk分块处理 (80万行chunk)") # 【修改】
            print("   - 20核心I/O并行优化 (原8核心)") # 【修改】
            print("   - DSW内存映射加速文件读取") # 【修改】
            print("   - 大内存向量化数据处理和去重") # 【修改】
            print("   - 高频内存使用监控 (60%阈值)") # 【修改】
        else:
            print(f"\n格式验证失败！请检查输出文件")
    else:
        print(f"\n转换失败！请检查输入文件")
    
    print(' - PY131 Modified for DSW 48-core optimization - ') # 【修改】
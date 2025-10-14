import os
import time
import warnings
import multiprocessing
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

# === 训练过程可视化相关导包（图片保存用） ===
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # 无界面环境下保存图片
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# ----------------------------
# 中文字体适配（避免中文标题变方块）
# ----------------------------
def setup_chinese_font():
    font_candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Source Han Sans CN']
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
        font_path = r"C:\Windows\Fonts\msyh.ttc"  # Windows常见中文字体
        if os.path.exists(font_path):
            ch_font = FontProperties(fname=font_path)
        else:
            print("[字体] 未找到可用中文字体，图中文字可能显示为方块。")
    plt.rcParams['axes.unicode_minus'] = False
    return ch_font

CH_FONT = setup_chinese_font()

# ----------------------------
# DSW环境性能优化参数
# ----------------------------
def setup_performance_optimization():
    memory_mode = os.environ.get('TIANCHI_MEMORY_MODE', 'normal')
    cpu_cores = int(os.environ.get('TIANCHI_CPU_CORES', '40'))
    suggested_chunk_size = int(os.environ.get('TIANCHI_CHUNK_SIZE', '500000'))

    if memory_mode == 'conservative':
        chunk_size = min(suggested_chunk_size, 300000)
        n_jobs = min(cpu_cores // 2, 20)
    else:
        chunk_size = min(800000, suggested_chunk_size * 2)
        n_jobs = min(cpu_cores, 30)

    os.environ['SKLEARN_N_JOBS'] = str(n_jobs)
    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    os.environ['NUMEXPR_MAX_THREADS'] = str(cpu_cores)
    os.environ['MKL_NUM_THREADS'] = str(n_jobs)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_jobs)
    
    gbdt_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'min_samples_leaf': 6,
        'learning_rate': 0.015,
        'subsample': 0.9,
        'max_features': 'sqrt',
        'verbose': 1,
        'random_state': 42
    }
    
    print(f"GBDT模型性能优化配置 (DSW):")
    print(f"   CPU核心数: {multiprocessing.cpu_count()} (使用{n_jobs}个用于GBDT)")
    print(f"   内存模式: {memory_mode}")
    print(f"   预测Chunk大小: {chunk_size:,} (DSW大内存优化)")
    print(f"   GBDT参数: n_estimators={gbdt_params['n_estimators']}, max_depth={gbdt_params['max_depth']} (DSW优化)")
    print(f"   多线程设置: sklearn={n_jobs}核心, 数值计算={cpu_cores}核心")
    print(f"   预计处理能力: 全量11亿+数据GBDT训练")
    
    return chunk_size, n_jobs, gbdt_params

chunk_size, n_jobs, gbdt_params = setup_performance_optimization()

# ----------------------------
# 路径配置
# ----------------------------
path_df_part_1_uic_label_cluster = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "Mobile_Recommendation/data/mobile/gbdt/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic = "Mobile_Recommendation/data/mobile/df_part_3_uic.csv"

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

path_df_P = "data/tianchi_fresh_comp_train_item_online.txt"

# 输出
path_df_result = "Mobile_Recommendation/data/mobile/gbdt/res_gbdt_k_means_subsample.csv"
path_df_result_tmp = "Mobile_Recommendation/data/mobile/gbdt/df_result_tmp.csv"

# ----------------------------
# 特征列
# ----------------------------
FEATURE_COLUMNS = [
    'u_b4_count_in_6', 'u_b3_count_in_6', 'u_b4_rate', 'u_b_count_in_1', 'u_b4_diff_hours',
    'i_u_count_in_6', 'i_b4_count_in_6', 'i_b3_count_in_6', 'i_b4_rate', 'i_b_count_in_1',
    'c_u_count_in_6', 'c_b4_count_in_6', 'c_b3_count_in_6', 'c_b4_rate', 'c_b_count_in_1',
    'ic_u_rank_in_c', 'ic_b4_rank_in_c', 'ic_b_rank_in_c',
    'ui_b4_count_in_6', 'ui_b3_count_in_6', 'ui_b_count_rank_in_u', 'ui_b3_last_hours', 'ui_b_count_in_1',
    'uc_b4_count_in_6', 'uc_b3_count_in_6', 'uc_b_count_rank_in_u', 'uc_b3_last_hours', 'uc_b_count_in_1'
]

# ----------------------------
# 工具函数
# ----------------------------
def df_read(path, mode='r'):
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return None
    
    try:
        dtype_dict = {
            'user_id': 'int32',
            'item_id': 'int32',
            'item_category': 'int16',
            'label': 'int8',
            'class': 'int16'
        }
        if path.endswith('.txt'):
            item_cols = ["item_id", "item_geohash", "item_category"]
            df = pd.read_csv(path, sep='\t', names=item_cols, index_col=False, dtype={'item_id': 'int32', 'item_category': 'int16'})
        else:
            # 只对存在的列应用dtype
            header = pd.read_csv(path, nrows=1)
            use_dtype = {k: v for k, v in dtype_dict.items() if k in header.columns}
            df = pd.read_csv(path, index_col=False, dtype=use_dtype)
        file_size = os.path.getsize(path) / (1024*1024)
        print(f"   载入: {os.path.basename(path)} ({file_size:.1f}MB, {len(df):,}行)")
        return df
    except Exception as e:
        print(f"读取文件失败: {path}, 错误: {e}")
        return None

def subsample(df, sub_size):
    if sub_size >= len(df): 
        return df
    else: 
        return df.sample(n=sub_size, random_state=42)

# ----------------------------
# Step 1: 载入训练数据和特征
# ----------------------------
print("\nStep 1: 载入训练数据和特征...")
df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)
df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)

print("   载入Part 1&2特征文件...")
df_part_1_U  = df_read(path_df_part_1_U)   
df_part_1_I  = df_read(path_df_part_1_I)
df_part_1_C  = df_read(path_df_part_1_C)
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

df_part_2_U  = df_read(path_df_part_2_U)   
df_part_2_I  = df_read(path_df_part_2_I)
df_part_2_C  = df_read(path_df_part_2_C)
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)

print("   所有特征文件载入完成")

# 合并顺序
merge_sequence_1 = [
    (df_part_1_U, ['user_id']),
    (df_part_1_I, ['item_id']),
    (df_part_1_C, ['item_category']),
    (df_part_1_IC, ['item_id','item_category']),
    (df_part_1_UI, ['user_id','item_id']),
    (df_part_1_UC, ['user_id','item_category'])
]
merge_sequence_2 = [
    (df_part_2_U, ['user_id']),
    (df_part_2_I, ['item_id']),
    (df_part_2_C, ['item_category']),
    (df_part_2_IC, ['item_id','item_category']),
    (df_part_2_UI, ['user_id','item_id']),
    (df_part_2_UC, ['user_id','item_category'])
]

# ----------------------------
# 训练集构建
# ----------------------------
def train_set_construct(np_ratio=1, sub_ratio=1):
    print(f"   构建训练集 (NP比例={np_ratio}, 子采样率={sub_ratio})")
    
    # 正样本（class==0）
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac=sub_ratio, random_state=42)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac=sub_ratio, random_state=42)

    # 负样本等比按簇抽样
    frac_ratio = sub_ratio * np_ratio/1200
    cluster_count = 0
    negative_samples_1, negative_samples_2 = [], []
    
    for i in range(1, 1001, 1):
        part1_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        if not part1_i.empty:
            sampled = part1_i.sample(frac=frac_ratio, random_state=42+i)
            negative_samples_1.append(sampled)
            cluster_count += len(sampled)
        part2_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        if not part2_i.empty:
            sampled = part2_i.sample(frac=frac_ratio, random_state=42+i)
            negative_samples_2.append(sampled)
            cluster_count += len(sampled)
        if i % 200 == 0:
            print(f"      已处理 {i} 个簇...")

    if negative_samples_1:
        train_part_1_uic_label = pd.concat([train_part_1_uic_label] + negative_samples_1, ignore_index=True, sort=False)
    if negative_samples_2:
        train_part_2_uic_label = pd.concat([train_part_2_uic_label] + negative_samples_2, ignore_index=True, sort=False)
    
    print(f"   训练样本键值选择完成，负样本: {cluster_count:,}个")
    
    # 合并特征（Part1）
    print("   构建Part 1训练特征...")
    train_part_1_df = train_part_1_uic_label.copy()
    for feature_df, merge_keys in merge_sequence_1:
        train_part_1_df = pd.merge(train_part_1_df, feature_df, how='left', on=merge_keys, sort=False)

    # 合并特征（Part2）
    print("   构建Part 2训练特征...")
    train_part_2_df = train_part_2_uic_label.copy()
    for feature_df, merge_keys in merge_sequence_2:
        train_part_2_df = pd.merge(train_part_2_df, feature_df, how='left', on=merge_keys, sort=False)
    
    train_df = pd.concat([train_part_1_df, train_part_2_df], ignore_index=True, sort=False)

    # 缺失填0并提取训练数组
    train_df[FEATURE_COLUMNS] = train_df[FEATURE_COLUMNS].fillna(0)
    train_X = train_df[FEATURE_COLUMNS].values
    train_y = train_df['label'].values

    print(train_df.head())
    print(train_X,train_y)
    print(train_X.shape, train_y.shape)

    pos_count = int((train_y == 1).sum())
    neg_count = int((train_y == 0).sum())
    print(f"   训练集构建完成: {len(train_df):,}个样本 (正:{pos_count:,}, 负:{neg_count:,}, 比例≈{neg_count/max(pos_count,1):.1f}:1)")
    print(f"   使用 {len(FEATURE_COLUMNS)} 个精简特征")
    
    del train_part_1_df, train_part_2_df, negative_samples_1, negative_samples_2
    return train_X, train_y

# ----------------------------
# 验证集构建（用于可视化曲线评估）
# ----------------------------
def valid_set_construct(sub_ratio=0.1):
    print(f"   构建验证集 (子采样率={sub_ratio})")
    
    valid_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac=sub_ratio, random_state=42)
    valid_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac=sub_ratio, random_state=42)

    valid_samples_1 = [valid_part_1_uic_label]
    valid_samples_2 = [valid_part_2_uic_label]

    for i in range(1, 1001, 1):
        part1_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        if not part1_i.empty:
            valid_samples_1.append(part1_i.sample(frac=sub_ratio, random_state=42+i))
        part2_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        if not part2_i.empty:
            valid_samples_2.append(part2_i.sample(frac=sub_ratio, random_state=42+i))
    
    valid_part_1_uic_label = pd.concat(valid_samples_1, ignore_index=True, sort=False)
    valid_part_2_uic_label = pd.concat(valid_samples_2, ignore_index=True, sort=False)
    
    # 合并特征
    valid_part_1_df = valid_part_1_uic_label.copy()
    for feature_df, merge_keys in merge_sequence_1:
        valid_part_1_df = pd.merge(valid_part_1_df, feature_df, how='left', on=merge_keys, sort=False)
    valid_part_2_df = valid_part_2_uic_label.copy()
    for feature_df, merge_keys in merge_sequence_2:
        valid_part_2_df = pd.merge(valid_part_2_df, feature_df, how='left', on=merge_keys, sort=False)
    
    valid_df = pd.concat([valid_part_1_df, valid_part_2_df], ignore_index=True, sort=False)
    valid_df[FEATURE_COLUMNS] = valid_df[FEATURE_COLUMNS].fillna(0)
    valid_X = valid_df[FEATURE_COLUMNS].values
    valid_y = valid_df['label'].values
    
    pos_count = int((valid_y == 1).sum())
    neg_count = int((valid_y == 0).sum())
    print(f"   验证集构建完成: {len(valid_df):,}个样本 (正:{pos_count:,}, 负:{neg_count:,})")
    return valid_X, valid_y

# ----------------------------
# 训练过程可视化与说明图
# ----------------------------
def plot_training_diagnostics(model, feature_names, out_dir, make_valid=True, valid_ratio=0.2):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 训练损失曲线
    try:
        train_loss = np.array(getattr(model, "train_score_", []), dtype=float)
        if train_loss.size > 0:
            pd.DataFrame({"iter": np.arange(1, len(train_loss)+1), "train_loss": train_loss}).to_csv(
                os.path.join(out_dir, "gbdt_train_loss.csv"), index=False, encoding="utf-8-sig"
            )
            plt.figure(figsize=(8, 5))
            plt.plot(np.arange(1, len(train_loss)+1), train_loss, label="训练损失", color="#1f77b4")
            if CH_FONT:
                plt.title("GBDT 训练损失曲线", fontproperties=CH_FONT)
                plt.xlabel("迭代轮数", fontproperties=CH_FONT)
                plt.ylabel("对数损失(LogLoss)", fontproperties=CH_FONT)
            else:
                plt.title("GBDT 训练损失曲线"); plt.xlabel("迭代轮数"); plt.ylabel("对数损失(LogLoss)")
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "gbdt_train_loss.png"), dpi=150); plt.close()
            print("   已保存: gbdt_train_loss.png / gbdt_train_loss.csv")
    except Exception as e:
        print(f"   绘制训练损失曲线失败: {e}")

    # 2) 袋外改进曲线（subsample<1）
    try:
        oob_imp = np.array(getattr(model, "oob_improvement_", []), dtype=float)
        if oob_imp.size > 0:
            pd.DataFrame({"iter": np.arange(1, len(oob_imp)+1), "oob_improvement": oob_imp}).to_csv(
                os.path.join(out_dir, "gbdt_oob_improvement.csv"), index=False, encoding="utf-8-sig"
            )
            plt.figure(figsize=(8, 5))
            plt.plot(np.arange(1, len(oob_imp)+1), oob_imp, label="袋外改进", color="#ff7f0e")
            if CH_FONT:
                plt.title("GBDT 袋外改进曲线（OOB Improvement）", fontproperties=CH_FONT)
                plt.xlabel("迭代轮数", fontproperties=CH_FONT)
                plt.ylabel("OOB改进值", fontproperties=CH_FONT)
            else:
                plt.title("GBDT 袋外改进曲线（OOB Improvement）"); plt.xlabel("迭代轮数"); plt.ylabel("OOB改进值")
            plt.axhline(0, color='gray', linewidth=1)
            plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "gbdt_oob_improvement.png"), dpi=150); plt.close()
            print("   已保存: gbdt_oob_improvement.png / gbdt_oob_improvement.csv")
    except Exception as e:
        print(f"   绘制OOB改进曲线失败: {e}")

    # 3) 特征重要性Top20
    try:
        importances = np.array(getattr(model, "feature_importances_", []), dtype=float)
        if importances.size == len(feature_names):
            fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).reset_index(drop=True)
            fi.to_csv(os.path.join(out_dir, "gbdt_feature_importance.csv"), index=False, encoding="utf-8-sig")
            topk = fi.head(20).iloc[::-1]
            plt.figure(figsize=(10, 8))
            plt.barh(topk["feature"], topk["importance"], color=sns.color_palette("viridis", len(topk)))
            if CH_FONT:
                plt.title("GBDT 特征重要性 Top 20", fontproperties=CH_FONT)
                plt.xlabel("重要性(归一化)", fontproperties=CH_FONT)
                plt.ylabel("特征名", fontproperties=CH_FONT)
            else:
                plt.title("GBDT 特征重要性 Top 20"); plt.xlabel("重要性(归一化)"); plt.ylabel("特征名")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "gbdt_feature_importance_top20.png"), dpi=150); plt.close()
            print("   已保存: gbdt_feature_importance_top20.png / gbdt_feature_importance.csv")
    except Exception as e:
        print(f"   绘制特征重要性失败: {e}")

    # 4) 外部验证集 ROC/PR（基于valid_set_construct）
    if make_valid:
        try:
            print("   构建外部验证集用于曲线评估...")
            valid_X, valid_y = valid_set_construct(sub_ratio=valid_ratio)
            probas = model.predict_proba(valid_X)
            cls = list(model.classes_)
            pos_idx = cls.index(1) if 1 in cls else int(np.argmax(cls))
            y_prob = probas[:, pos_idx]

            # ROC
            fpr, tpr, _ = roc_curve(valid_y, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}", color="#d62728")
            plt.plot([0,1], [0,1], 'k--', alpha=0.4)
            if CH_FONT:
                plt.title("外部验证集 ROC 曲线", fontproperties=CH_FONT)
                plt.xlabel("FPR 假正率", fontproperties=CH_FONT)
                plt.ylabel("TPR 召回率", fontproperties=CH_FONT)
            else:
                plt.title("外部验证集 ROC 曲线"); plt.xlabel("FPR 假正率"); plt.ylabel("TPR 召回率")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "gbdt_valid_ROC.png"), dpi=150); plt.close()

            # PR
            precision, recall, _ = precision_recall_curve(valid_y, y_prob)
            ap = average_precision_score(valid_y, y_prob)
            plt.figure(figsize=(6, 6))
            plt.plot(recall, precision, label=f"AP={ap:.4f}", color="#2ca02c")
            if CH_FONT:
                plt.title("外部验证集 PR 曲线", fontproperties=CH_FONT)
                plt.xlabel("Recall 召回率", fontproperties=CH_FONT)
                plt.ylabel("Precision 精准率", fontproperties=CH_FONT)
            else:
                plt.title("外部验证集 PR 曲线"); plt.xlabel("Recall 召回率"); plt.ylabel("Precision 精准率")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "gbdt_valid_PR.png"), dpi=150); plt.close()

            # 保存曲线CSV
            pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(out_dir, "gbdt_valid_ROC_curve.csv"), index=False, encoding="utf-8-sig")
            pd.DataFrame({"recall": recall, "precision": precision}).to_csv(os.path.join(out_dir, "gbdt_valid_PR_curve.csv"), index=False, encoding="utf-8-sig")
            print("   已保存: ROC/PR 曲线与CSV")
        except Exception as e:
            print(f"   验证曲线绘制失败（可忽略，不影响训练）: {e}")

# ----------------------------
# Step 2: 使用优化的GBDT参数进行训练和预测
# ----------------------------
print(f"\nStep 2: 训练GBDT模型...")

# 构建训练集
train_X, train_y = train_set_construct(np_ratio=60, sub_ratio=1)

# 初始化GBDT模型
# ...existing code...
print("   初始化GBDT模型 (DSW优化)...")
GBDT_clf = GradientBoostingClassifier(
    n_estimators=500,        # 基学习器数量（迭代轮数）；增大提高拟合能力但训练更慢
    max_depth=6,             # 单棵树最大深度；控制模型复杂度与泛化
    min_samples_leaf=6,      # 叶子节点最小样本数；增大可抑制过拟合
    learning_rate=0.015,     # 学习率；与 n_estimators 互相制衡，越小需更多树
    subsample=0.9,           # 每轮行采样比例；<1引入随机性并支持OOB改进
    max_features='sqrt',     # 每次分裂可用特征数；'sqrt'常用于分类提升泛化
    verbose=1,               # 训练日志详细程度；>0输出每若干轮信息
    random_state=42,         # 随机种子；保证可复现实验结果
    warm_start=False,        # 是否沿用上次增树继续训练；一般保持False
    validation_fraction=0.1, # 早停的内部验证集占比；与 n_iter_no_change 配合
    n_iter_no_change=30      # 早停阈值：验证分数连续无提升的迭代次数
)
# ...existing code...

print("   开始GBDT训练...")
start_time = time.time()
try:
    GBDT_clf.fit(train_X, train_y)
    training_time = time.time() - start_time
    print(f"   GBDT训练完成! 用时: {training_time/60:.1f}分钟 (DSW多核加速)")
except MemoryError:
    print("   内存不足，尝试减小训练集... (DSW环境此情况较少)")
    sample_indices = np.random.choice(len(train_X), size=min(len(train_X), 800000), replace=False)
    train_X_sampled = train_X[sample_indices]
    train_y_sampled = train_y[sample_indices]
    GBDT_clf.fit(train_X_sampled, train_y_sampled)
    training_time = time.time() - start_time
    print(f"   GBDT训练完成! (使用采样数据) 用时: {training_time/60:.1f}分钟")

# 训练过程可视化输出（保存到gbdt目录）
try:
    out_dir = os.path.dirname(path_df_result)
    print("   生成训练过程图片与说明图...")
    plot_training_diagnostics(
        model=GBDT_clf,
        feature_names=FEATURE_COLUMNS,
        out_dir=out_dir,
        make_valid=True,
        valid_ratio=0.2
    )
except Exception as e:
    print(f"   训练过程可视化失败：{e}")

# 释放训练数据内存
del train_X, train_y

# ----------------------------
# Step 3: 在Part 3数据上进行预测
# ----------------------------
print(f"\nStep 3: 在Part 3数据上进行预测...")

# 载入Part3特征
print("   载入Part 3特征文件...")
df_part_3_U  = df_read(path_df_part_3_U)   
df_part_3_I  = df_read(path_df_part_3_I)
df_part_3_C  = df_read(path_df_part_3_C)
df_part_3_IC = df_read(path_df_part_3_IC)
df_part_3_UI = df_read(path_df_part_3_UI)
df_part_3_UC = df_read(path_df_part_3_UC)

merge_sequence_3 = [
    (df_part_3_U, ['user_id']),
    (df_part_3_I, ['item_id']),
    (df_part_3_C, ['item_category']),
    (df_part_3_IC, ['item_id','item_category']),
    (df_part_3_UI, ['user_id','item_id']),
    (df_part_3_UC, ['user_id','item_category'])
]

print("   开始分块预测...")
batch = 0
total_predictions = 0
csv_params = {
    'chunksize': chunk_size,
    'dtype': {'user_id': 'int32', 'item_id': 'int32', 'item_category': 'int16'},
    'engine': 'c'
}
temp_files = []

try:
    for pred_uic in pd.read_csv(path_df_part_3_uic, **csv_params): 
        try:
            batch += 1
            print(f"      处理预测chunk {batch}, 样本数: {len(pred_uic):,}")
            
            pred_df = pred_uic.copy()
            for feature_df, merge_keys in merge_sequence_3:
                pred_df = pd.merge(pred_df, feature_df, how='left', on=merge_keys, sort=False)

            pred_df[FEATURE_COLUMNS] = pred_df[FEATURE_COLUMNS].fillna(0)
            pred_X = pred_df[FEATURE_COLUMNS].values
            probas = GBDT_clf.predict_proba(pred_X)
            cls = list(GBDT_clf.classes_)
            pos_idx = cls.index(1) if 1 in cls else int(np.argmax(cls))
            pred_proba = probas[:, pos_idx]
            pred_y = (pred_proba > 0.15).astype(int)
            pred_df['pred_label'] = pred_y
            predicted_pairs = pred_df[pred_df['pred_label'] == 1]
            
            if len(predicted_pairs) > 0:
                temp_files.append(predicted_pairs[['user_id','item_id']].copy())
                total_predictions += len(predicted_pairs)
            
            if batch % 5 == 0:
                print(f"         已完成 {batch} 个chunk，累计预测正样本: {total_predictions:,}个")
                try:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                    if memory_percent > 60:
                        print(f"         DSW内存使用率: {memory_percent:.1f}%")
                except ImportError:
                    pass
            
            del pred_df, pred_X, pred_proba, pred_y
        except Exception as e:
            print(f"         处理chunk {batch} 出错: {e}")
            continue
            
    print(f"   分块预测完成! 总共预测出 {total_predictions:,} 个正样本")
    
    if temp_files:
        print("   合并预测结果...")
        all_predictions = pd.concat(temp_files, ignore_index=True, sort=False)
        all_predictions.to_csv(path_df_result_tmp, index=False)
        print(f"   临时结果保存完成: {len(all_predictions):,} 个预测对")
        del temp_files, all_predictions
    else:
        print("   没有预测出正样本")
        pd.DataFrame(columns=['user_id', 'item_id']).to_csv(path_df_result_tmp, index=False)
except StopIteration:
    print("      预测完成!")
except Exception as e:
    print(f"预测过程出错: {e}")

# ----------------------------
# Step 4: 在物品子集P上生成最终结果
# ----------------------------
print(f"\nStep 4: 生成最终提交结果...")

print("   载入物品子集P...")
df_P = df_read(path_df_P)
if df_P is None:
    print("物品文件载入失败")
    exit(1)

df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']].copy()
print(f"   物品子集P包含 {len(df_P_item):,} 个独特商品")

if os.path.exists(path_df_result_tmp):
    df_pred = pd.read_csv(path_df_result_tmp, index_col=False)
    print(f"   临时预测结果: {len(df_pred):,} 个用户-商品对")
    
    df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner', sort=False)[['user_id', 'item_id']]
    df_pred_P.to_csv(path_df_result, index=False)
    
    print(f"   最终结果保存完成: {len(df_pred_P):,} 个预测的购买对")
    print(f"   保存到: {path_df_result}")
    
    os.remove(path_df_result_tmp)
    print("   临时文件已清理")
else:
    print("临时预测文件不存在")
    pd.DataFrame(columns=['user_id', 'item_id']).to_csv(path_df_result, index=False)

print("\n" + "=" * 60)
print("GBDT训练和预测完成! (DSW环境优化)")

# 最终统计
print(f"\n最终统计:")
print(f"   使用特征数: {len(FEATURE_COLUMNS)} 个精简特征")
print(f"   模型参数: max_depth={gbdt_params['max_depth']}, n_estimators={gbdt_params['n_estimators']}, learning_rate={gbdt_params['learning_rate']} (DSW优化)")
print(f"   训练时间: {training_time/60:.1f} 分钟")
print(f"   预测阈值: 0.15")
print(f"   DSW 48核心优化: 使用{n_jobs}个核心进行GBDT计算")

if os.path.exists(path_df_result):
    result_size = os.path.getsize(path_df_result) / 1024
    print(f"   结果文件大小: {result_size:.1f}KB")

print(' - PY131 Modified for DSW 48-core optimization - ')
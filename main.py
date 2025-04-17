import sys
import os
import io
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras import mixed_precision
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models, callbacks

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score
)
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTENC

# Matplotlib 中文字體設定
try:
    mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    mpl.rcParams['axes.unicode_minus'] = False
    print("[INFO] 已設定 Matplotlib 字體為 Microsoft JhengHei")
except Exception as e:
    print(f"[WARN] 設定字體失敗: {e} - 可能無法顯示中文")

# 效能設定 (GPU + 混合精度 + XLA)
def setup_performance():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] 偵測到 {len(gpus)} 個 GPU: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] 已為 GPU 設定記憶體動態增長")
        except RuntimeError as e:
            print(f"[WARN] 設定 GPU 記憶體增長失敗: {e}")
        # 啟用混合精度
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"[INFO] 已啟用混合精度: {mixed_precision.global_policy().name}")
    else:
        print("[WARN] 未偵測到 GPU, 將使用 CPU 訓練.")
    # 嘗試啟用 XLA JIT
    try:
        tf.config.optimizer.set_jit(True)
        print("[INFO] 已嘗試啟用 XLA JIT 編譯")
    except Exception as e:
        print(f"[WARN] 啟用 XLA 時錯誤: {e}")

# 讀取資料
def load_data(filepath):
    print(f"[INFO] 讀取資料: {filepath}")
    try:
        df = pd.read_csv(filepath)
        print("[INFO] 資料載入成功!")
        return df
    except FileNotFoundError:
        print(f"[ERROR] 檔案不存在: {filepath}")
        sys.exit(1)

# 執行探索性資料分析 (EDA)
def perform_eda(df, target="Is_laundering", save_plots=False, plot_dir="plots"):
    print("[INFO] EDA 開始")

    # 基本資訊
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())
    print("[INFO] DataFrame 前五筆：\n", df.head())
    print("[INFO] 數值統計：\n", df.describe())

    # 缺失值
    missing_sum = df.isnull().sum()
    if missing_sum.any():
        print("\n[INFO] 缺失值狀況：\n", missing_sum[missing_sum > 0])
    else:
        print("\n[INFO] 目前無缺失值")

    # 目標變數分佈
    if target in df.columns:
        print("\n--- 標籤分布 ---\n", df[target].value_counts())
        plt.figure(figsize=(6,4))
        sns.countplot(x=target, data=df)
        plt.title("Label Distribution")
        plt.tight_layout()
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
            fpath = os.path.join(plot_dir, "label_distribution.png")
            plt.savefig(fpath)
            print("[INFO] 已儲存圖:", fpath)
        else:
            plt.show()
        plt.close()
    else:
        print(f"[WARN] df 中無 {target} 欄位.")

    # 相關係數熱圖 (若維度不大)
    numeric_cols_count = df.select_dtypes(include=[np.number]).shape[1]
    if numeric_cols_count > 0 and numeric_cols_count < 20:
        plt.figure(figsize=(8,6))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
            fpath = os.path.join(plot_dir, "correlation_heatmap.png")
            plt.savefig(fpath)
            print("[INFO] 已儲存相關係數熱圖:", fpath)
        else:
            plt.show()
        plt.close()

    # 數值欄位盒鬚圖
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plt.figure(figsize=(10, 5))
        for idx, col in enumerate(numeric_cols):
            plt.subplot(1, len(numeric_cols), idx+1)
            sns.boxplot(data=df, y=col)
            plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
            fpath = os.path.join(plot_dir, "boxplot_numeric.png")
            plt.savefig(fpath)
            print("[INFO] 已儲存盒鬚圖:", fpath)
        else:
            plt.show()
        plt.close()

    print("[INFO] EDA 結束")

# 特徵工程
def prepare_features(df, target="Is_laundering"):
    """
    處理步驟:
    1) 清理缺失值
    2) 移除極端值 (Amount)
    3) 移除不必要欄位
    4) Log transform + StandardScaler (Amount)
    5) Label Encoding 類別變數
    """
    numeric_cols = ["Amount"]
    cat_cols = [
        "Payment_currency", "Payment_type", "Sender_bank_location",
        "Receiver_bank_location", "Received_currency"
    ]
    drop_cols = ["Time", "Date", "Sender_account", "Receiver_account", "Laundering_type"]

    df.dropna(inplace=True)

    # 移除 Amount 極端值 (1% ~ 99%)
    if "Amount" in df.columns:
        Q1 = df["Amount"].quantile(0.01)
        Q3 = df["Amount"].quantile(0.99)
        df = df[(df["Amount"] >= Q1) & (df["Amount"] <= Q3)]

    # 移除欄位
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    if target not in df.columns:
        print(f"[ERROR] 缺少目標欄位: {target}")
        sys.exit(1)

    X = df.drop(columns=[target])
    y = df[target]

    # Log1p 轉換 Amount
    if "Amount" in X.columns:
        X["Amount"] = np.log1p(X["Amount"].values)

    # LabelEncoding 類別特徵
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # StandardScaler 數值特徵
    for col in numeric_cols:
        if col in X.columns:
            sc = StandardScaler()
            X[col] = sc.fit_transform(X[[col]])

    return X, y, numeric_cols, cat_cols

# 建立 Autoencoder 模型
def build_autoencoder(input_dim, encoding_dim=16):
    inputs = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(inputs)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    decoded = layers.Dense(64, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    autoencoder = models.Model(inputs, decoded, name="autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# 訓練 Autoencoder (使用正常樣本)
def train_autoencoder(X_normal, epochs=20, batch_size=64):
    print("[INFO] 訓練 Autoencoder(無監督) ...")
    input_dim = X_normal.shape[1]
    autoencoder = build_autoencoder(input_dim=input_dim, encoding_dim=16)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = autoencoder.fit(
        X_normal, X_normal,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.2,
        verbose=1,
        callbacks=[es]
    )
    return autoencoder, history

# 使用 Autoencoder 偵測異常 (計算重建誤差)
def detect_anomalies_autoencoder(autoencoder, X_all):
    recon = autoencoder.predict(X_all)
    errors = np.mean((X_all - recon)**2, axis=1)
    # 使用 mean + 2*std 作為閾值
    threshold = np.mean(errors) + 2.0 * np.std(errors)
    print(f"[INFO] Autoencoder 重建誤差閾值: {threshold:.6f}")
    suspicious_mask = (errors > threshold)
    return suspicious_mask, errors

# Focal Loss 函數
def focal_loss(gamma=2.0, alpha=0.2):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        # 計算 cross entropy
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true)*tf.math.log(1 - y_pred)
        # 計算 focal loss weight
        weight = alpha*y_true + (1 - alpha)*(1 - y_true)
        focal_weight = tf.pow((1 - y_pred), gamma)
        # 計算最終 loss
        loss = tf.reduce_mean(focal_weight * weight * ce)
        return loss
    return loss_fn

# 建立使用 Focal Loss 的 DNN 模型
def build_focal_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid') # 二元分類輸出
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=focal_loss(gamma=2.0, alpha=0.2), # 使用 Focal Loss
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(name="AUC"),
            tf.keras.metrics.AUC(name="PRC", curve="PR"), # Precision-Recall AUC
        ]
    )
    return model

# 使用 SMOTENC 處理不平衡資料 (包含類別特徵)
def smote_balance(X, y, cat_cols):
    print("[INFO] SMOTENC 處理中...")
    # 取得類別特徵的索引
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    smotenc = SMOTENC(categorical_features=cat_idx, random_state=777)
    X_res, y_res = smotenc.fit_resample(X, y)
    print("[INFO] 過採樣後分布:\n", pd.Series(y_res).value_counts())
    return X_res, y_res

# 繪製並儲存混淆矩陣
def plot_and_save_cm(cm, title="Confusion Matrix", plot_dir="plots", save_plots=False, filename="cm.png"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, filename)
        plt.savefig(fpath)
        print("[INFO] 已儲存圖:", fpath)
    else:
        plt.show()
    plt.close()

# 繪製並儲存 Precision-Recall 曲線
def plot_and_save_prc(recall, precision, plot_dir="plots", save_plots=False, filename="pr_curve.png"):
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
        fpath = os.path.join(plot_dir, filename)
        plt.savefig(fpath)
        print("[INFO] 已儲存圖:", fpath)
    else:
        plt.show()
    plt.close()

# 主程式: 整合 Autoencoder(無監督) + Focal Loss DNN(監督) 二階段
def main():
    # --- 參數設定 ---
    filepath = "data/SAML-D.csv"
    target_col = "Is_laundering"
    save_plots = True
    plot_dir = "eda_plots_tf"
    save_model = True
    model_path = "saved_tf_model.keras"

    # --- 初始化 ---
    setup_performance()

    # --- 1. 讀取資料 + EDA ---
    df = load_data(filepath)
    perform_eda(df, target=target_col, save_plots=save_plots, plot_dir=plot_dir)

    # --- 2. 特徵工程 ---
    X_all, y_all, num_cols, cat_cols = prepare_features(df, target=target_col)

    # 分離正常樣本 (用於 Autoencoder 訓練)
    normal_mask = (y_all == 0)
    X_normal = X_all[normal_mask]
    X_sus = X_all[~normal_mask] # 已知洗錢樣本 (僅供參考)
    print(f"[INFO] 正常樣本數: {len(X_normal)}, 已知洗錢樣本數: {len(X_sus)}")

    # --- 階段 A: Autoencoder 無監督偵測 ---
    autoencoder, history_ae = train_autoencoder(X_normal, epochs=20, batch_size=64)
    suspicious_mask, errors = detect_anomalies_autoencoder(autoencoder, X_all)
    print("[INFO] Autoencoder 認定可疑筆數:", suspicious_mask.sum())

    # (可選) 評估 Autoencoder 偵測成效 (若有真實標籤)
    if 1 in y_all.values:
        oc_preds_bin = suspicious_mask.astype(int)
        print("\n=== [Autoencoder 初步結果] ===")
        print(classification_report(y_all, oc_preds_bin, digits=4))
        cm = confusion_matrix(y_all, oc_preds_bin)
        plot_and_save_cm(cm, title="Autoencoder CM", plot_dir=plot_dir,
                         save_plots=save_plots, filename="autoencoder_cm.png")

    # --- 階段 B: 合併標籤 (Autoencoder 偵測 + 原始標籤) ---
    two_stage_label = np.zeros_like(y_all.values)
    two_stage_label[suspicious_mask] = 1 # Autoencoder 標記為可疑
    two_stage_label[y_all == 1] = 1      # 原始標記為洗錢
    print(f"[INFO] 二階段資料中, label=1 的筆數: {(two_stage_label == 1).sum()}")
    print(f"[INFO] label=0 的筆數: {(two_stage_label == 0).sum()}")

    # --- 階段 C: 監督式學習 (Focal Loss DNN) ---
    # 使用二階段標籤分割資料
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, two_stage_label,
        test_size=0.3,
        random_state=777,
        stratify=two_stage_label # 確保分層抽樣
    )
    print("[INFO] 二階段樣本: 訓練=", len(X_tr), "測試=", len(X_te))

    # 對訓練集進行 SMOTENC 過採樣
    X_res, y_res = smote_balance(X_tr, y_tr, cat_cols)

    # 建立並訓練 Focal Loss DNN 模型
    model = build_focal_model(input_dim=X_res.shape[1])
    print(model.summary())
    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_res, y_res,
        epochs=40,
        batch_size=64,
        validation_split=0.2,
        callbacks=[es],
        verbose=1
    )

    # --- 模型評估 (使用測試集) ---
    y_pred_prob = model.predict(X_te).flatten()
    # 使用閾值 0.9 進行二元分類 (可調整)
    y_pred_bin = (y_pred_prob > 0.9).astype(int)

    print("\n=== [二階段監督模型 分類報告] ===")
    print(classification_report(y_te, y_pred_bin, digits=4))

    # 繪製混淆矩陣
    cm2 = confusion_matrix(y_te, y_pred_bin)
    plot_and_save_cm(cm2, title="2-Stage Confusion Matrix",
                     plot_dir=plot_dir, save_plots=save_plots,
                     filename="two_stage_cm.png")

    # 繪製 Precision-Recall 曲線
    precision, recall, _ = precision_recall_curve(y_te, y_pred_prob)
    plot_and_save_prc(recall, precision, plot_dir=plot_dir,
                      save_plots=save_plots, filename="two_stage_prc.png")

    # 計算 ROC AUC
    auc_val = roc_auc_score(y_te, y_pred_prob)
    print(f"[INFO] 2階段模型 ROC AUC = {auc_val:.4f}")

    # --- (可選) 儲存模型 ---
    if save_model:
        model.save(model_path)
        print(f"[INFO] 已儲存二階段監督模型至: {model_path}")

    print("[INFO] 程式執行完畢!")

if __name__ == "__main__":
    main()
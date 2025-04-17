# 使用 Autoencoder 與 Focal Loss DNN 進行反洗錢偵測 (AML Detection)

本專案旨在利用機器學習技術偵測潛在的洗錢交易。採用了一個二階段的方法：

1.  **無監督學習階段：** 使用 Autoencoder 模型對所有交易資料進行重建，計算重建誤差。高重建誤差的交易被初步標記為可疑。此階段利用正常交易模式來識別異常。
2.  **監督學習階段：** 結合 Autoencoder 偵測到的可疑交易與原始資料中已知的洗錢標籤，形成一個新的標籤集。接著，使用這些標籤訓練一個深度神經網路 (DNN)。為了處理資料不平衡問題，訓練過程中使用了 SMOTENC 進行過採樣，並採用 Focal Loss 作為損失函數，讓模型更專注於難以分類的樣本。

## 主要功能

*   **資料讀取與探索性資料分析 (EDA):** 載入 CSV 資料，進行基本資訊檢視、缺失值檢查、目標變數分佈視覺化、相關性分析和數值特徵分佈視覺化。
*   **特徵工程:**
    *   處理缺失值。
    *   移除交易金額 (`Amount`) 的極端值。
    *   移除不必要的欄位 (如時間、帳戶)。
    *   對 `Amount` 進行 Log1p 轉換和標準化 (StandardScaler)。
    *   對類別特徵進行標籤編碼 (LabelEncoding)。
*   **Autoencoder 模型:**
    *   建立並訓練 Autoencoder 模型 (僅使用正常交易資料)。
    *   使用訓練好的 Autoencoder 計算所有資料的重建誤差，並設定閾值以偵測異常交易。
*   **Focal Loss DNN 模型:**
    *   實作 Focal Loss 函數以處理類別不平衡。
    *   建立使用 Focal Loss 的 DNN 分類模型。
*   **資料不平衡處理:** 使用 SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous) 對訓練資料進行過採樣，以平衡正負樣本比例。
*   **模型訓練與評估:**
    *   使用合併後的標籤 (Autoencoder 偵測 + 原始標籤) 和 SMOTENC 處理後的資料訓練 DNN 模型。
    *   使用獨立的測試集評估模型效能，指標包含分類報告 (Precision, Recall, F1-score)、混淆矩陣、Precision-Recall 曲線和 ROC AUC 分數。
*   **效能優化:**
    *   支援 GPU 訓練。
    *   啟用 TensorFlow 混合精度 (Mixed Precision) 以加速訓練並減少記憶體使用。
    *   嘗試啟用 XLA (Accelerated Linear Algebra) JIT 編譯。
*   **視覺化:** 繪製並儲存目標分佈、相關係數熱圖、盒鬚圖、混淆矩陣和 PR 曲線。
*   **模型儲存:** 可選擇將訓練好的 DNN 模型儲存為 `.keras` 檔案。

## 環境需求

*   Python 3.x
*   主要的 Python 函式庫：
    *   TensorFlow (>= 2.x)
    *   Scikit-learn
    *   Pandas
    *   NumPy
    *   Matplotlib
    *   Seaborn
    *   Imbalanced-learn

## 安裝

1.  複製此儲存庫：
    ```bash
    git clone https://github.com/your-username/AML_Model.git # 請替換成您的儲存庫 URL
    cd AML_Model
    ```
2.  (建議) 建立並啟用虛擬環境：
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    # source venv/bin/activate
    ```
3.  安裝所需的套件 (建議建立 `requirements.txt` 檔案)：
    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```
    或者，如果您有 `requirements.txt` 檔案：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1.  **準備資料:** 將您的交易資料 CSV 檔案放置在 `data/` 資料夾下，並確保其名稱為 `SAML-D.csv` (或修改 `main.py` 中的 `filepath` 變數)。資料應包含 `Is_laundering` 作為目標欄位，以及程式碼中 `prepare_features` 函數所使用的相關特徵欄位。
2.  **執行腳本:**
    ```bash
    python main.py
    ```
3.  **查看結果:**
    *   程式執行過程中的資訊和評估結果會輸出到終端機。
    *   如果 `save_plots` 設為 `True`，相關的 EDA 圖表和模型評估圖表會儲存在 `eda_plots_tf/` 資料夾 (或 `plot_dir` 指定的資料夾) 中。
    *   如果 `save_model` 設為 `True`，訓練好的 DNN 模型會儲存為 `saved_tf_model.keras` (或 `model_path` 指定的路徑)。

## 設定調整

您可以在 `main.py` 的 `main()` 函數開頭調整以下參數：

*   `filepath`: 輸入資料檔案的路徑。
*   `target_col`: 目標變數 (標籤) 的欄位名稱。
*   `save_plots`: 是否儲存產生的圖表。
*   `plot_dir`: 儲存圖表的資料夾名稱。
*   `save_model`: 是否儲存訓練好的模型。
*   `model_path`: 儲存模型的檔案路徑。
*   Autoencoder 和 DNN 的訓練週期 (`epochs`)、批次大小 (`batch_size`) 等超參數也可在對應的訓練函數呼叫中調整。
*   `detect_anomalies_autoencoder` 中的閾值計算方式。
*   `main()` 函數中模型評估的二元分類閾值 (目前為 `0.9`)。

## 授權 (License)

(可選) 請在此處添加您的專案授權資訊，例如 MIT License, Apache License 2.0 等。
```# 使用 Autoencoder 與 Focal Loss DNN 進行反洗錢偵測 (AML Detection)

本專案旨在利用機器學習技術偵測潛在的洗錢交易。採用了一個二階段的方法：

1.  **無監督學習階段：** 使用 Autoencoder 模型對所有交易資料進行重建，計算重建誤差。高重建誤差的交易被初步標記為可疑。此階段利用正常交易模式來識別異常。
2.  **監督學習階段：** 結合 Autoencoder 偵測到的可疑交易與原始資料中已知的洗錢標籤，形成一個新的標籤集。接著，使用這些標籤訓練一個深度神經網路 (DNN)。為了處理資料不平衡問題，訓練過程中使用了 SMOTENC 進行過採樣，並採用 Focal Loss 作為損失函數，讓模型更專注於難以分類的樣本。

## 主要功能

*   **資料讀取與探索性資料分析 (EDA):** 載入 CSV 資料，進行基本資訊檢視、缺失值檢查、目標變數分佈視覺化、相關性分析和數值特徵分佈視覺化。
*   **特徵工程:**
    *   處理缺失值。
    *   移除交易金額 (`Amount`) 的極端值。
    *   移除不必要的欄位 (如時間、帳戶)。
    *   對 `Amount` 進行 Log1p 轉換和標準化 (StandardScaler)。
    *   對類別特徵進行標籤編碼 (LabelEncoding)。
*   **Autoencoder 模型:**
    *   建立並訓練 Autoencoder 模型 (僅使用正常交易資料)。
    *   使用訓練好的 Autoencoder 計算所有資料的重建誤差，並設定閾值以偵測異常交易。
*   **Focal Loss DNN 模型:**
    *   實作 Focal Loss 函數以處理類別不平衡。
    *   建立使用 Focal Loss 的 DNN 分類模型。
*   **資料不平衡處理:** 使用 SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous) 對訓練資料進行過採樣，以平衡正負樣本比例。
*   **模型訓練與評估:**
    *   使用合併後的標籤 (Autoencoder 偵測 + 原始標籤) 和 SMOTENC 處理後的資料訓練 DNN 模型。
    *   使用獨立的測試集評估模型效能，指標包含分類報告 (Precision, Recall, F1-score)、混淆矩陣、Precision-Recall 曲線和 ROC AUC 分數。
*   **效能優化:**
    *   支援 GPU 訓練。
    *   啟用 TensorFlow 混合精度 (Mixed Precision) 以加速訓練並減少記憶體使用。
    *   嘗試啟用 XLA (Accelerated Linear Algebra) JIT 編譯。
*   **視覺化:** 繪製並儲存目標分佈、相關係數熱圖、盒鬚圖、混淆矩陣和 PR 曲線。
*   **模型儲存:** 可選擇將訓練好的 DNN 模型儲存為 `.keras` 檔案。

## 環境需求

*   Python 3.x
*   主要的 Python 函式庫：
    *   TensorFlow (>= 2.x)
    *   Scikit-learn
    *   Pandas
    *   NumPy
    *   Matplotlib
    *   Seaborn
    *   Imbalanced-learn

## 安裝

1.  複製此儲存庫：
    ```bash
    git clone https://github.com/your-username/AML_Model.git # 請替換成您的儲存庫 URL
    cd AML_Model
    ```
2.  (建議) 建立並啟用虛擬環境：
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    # source venv/bin/activate
    ```
3.  安裝所需的套件 (建議建立 `requirements.txt` 檔案)：
    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
    ```
    或者，如果您有 `requirements.txt` 檔案：
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

1.  **準備資料:** 將您的交易資料 CSV 檔案放置在 `data/` 資料夾下，並確保其名稱為 `SAML-D.csv` (或修改 `main.py` 中的 `filepath` 變數)。資料應包含 `Is_laundering` 作為目標欄位，以及程式碼中 `prepare_features` 函數所使用的相關特徵欄位。
2.  **執行腳本:**
    ```bash
    python main.py
    ```
3.  **查看結果:**
    *   程式執行過程中的資訊和評估結果會輸出到終端機。
    *   如果 `save_plots` 設為 `True`，相關的 EDA 圖表和模型評估圖表會儲存在 `eda_plots_tf/` 資料夾 (或 `plot_dir` 指定的資料夾) 中。
    *   如果 `save_model` 設為 `True`，訓練好的 DNN 模型會儲存為 `saved_tf_model.keras` (或 `model_path` 指定的路徑)。

## 設定調整

您可以在 `main.py` 的 `main()` 函數開頭調整以下參數：

*   `filepath`: 輸入資料檔案的路徑。
*   `target_col`: 目標變數 (標籤) 的欄位名稱。
*   `save_plots`: 是否儲存產生的圖表。
*   `plot_dir`: 儲存圖表的資料夾名稱。
*   `save_model`: 是否儲存訓練好的模型。
*   `model_path`: 儲存模型的檔案路徑。
*   Autoencoder 和 DNN 的訓練週期 (`epochs`)、批次大小 (`batch_size`) 等超參數也可在對應的訓練函數呼叫中調整。
*   `detect_anomalies_autoencoder` 中的閾值計算方式。
*   `main()` 函數中模型評估的二元分類閾值 (目前為 `0.9`)。


\---

applyTo: "preprocess/**"

\---

# 背景
RK3588 離線中文路牌警告語辨識系統，本模組是 OCR 之前的畫質改善（實驗性、可關閉）。

\# preprocess 模組（畫質改善，後續新增的實驗性算法）



\## 定位

這是可選的實驗模組，不是管線必要元件。

\- 預設 mode="passthrough"：直接回傳輸入的最後一幀 ROI，行為等同沒有此模組。

\- 其他 mode 皆由 config 切換，不得修改其他模組的程式碼。

\- 此模組失敗（例外、逾時、對位失敗）時必須退回 passthrough，不可讓管線中斷。



\## 介面（與其他模組的契約，不可擅自更改）

輸入：list\[TrackFrame]（frame\_idx, timestamp, full\_frame, bbox），時間戳不等間隔

輸出：PreprocessResult(image: BGR uint8, diagnostics: dict)



\## 演算法清單（各自獨立檔案，可單獨測試）

\- passthrough / best\_frame（挑最清晰幀，清晰度不可只用 Laplacian variance）

\- enhance：Gamma、CLAHE（只作用於 LAB 的 L 通道）

\- fusion：整幀對位（ECC, homography）→ 平均或中位數融合



\## 約束

\- 順序：對位 → 融合 → 對比增強。不可先增強再平均。

\- 在整幀上估計運動再裁切，不可先裁 bbox 再對位。

\- 最大融合幀數 N 與時間預算皆為 config 參數；不假設固定幀率。

\- 核心程式只用 numpy、opencv；實驗與評估程式放 `preprocess/eval/`，可用 scikit-image。

\- 輸入尺寸不固定（ROI 可能被放大），不可假設固定大小。

\- 每個函式回報耗時（ms）到 diagnostics。



\## 測試

\- 用合成退化影像做 pytest，不依賴外部資料集。

\- 每個 mode 至少一個測試，並包含「失敗時退回 passthrough」的測試。



\## 評估（與 baseline 對照）

mode 切換即為對照組：passthrough、best\_frame、enhance、fusion。

主指標是下游 OCR 的 CER，其次是延遲。合成資料才算 PSNR/SSIM。

負面結果（融合沒贏 baseline）要如實記錄。


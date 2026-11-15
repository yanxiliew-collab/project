# 原始資訊上限診斷：`data/road.mp4`

## 一頁摘要

- 診斷範圍：原始 frame 350–385；來源/edge spread 以原始 3840×2160 影格像素量測，OCR 控制組另明確標註 canonical、bicubic 輸入與既有 pipeline 輸出。
- 路牌 ROI：每幀 HSV 候選框，完整座標與數值見 `diagnosis.json`；靜態對照 ROI：`[0, 400, 280, 1300]`（左側水泥柱/邊界區域）。
- ffprobe：codec=`h264`, pix_fmt=`yuv420p`, bitrate=`25351926.0` bps，bits/pixel/frame=`0.10188370466820988`。
- frame 350–385 的 I/P/B 分佈：`{'B': 1, 'P': 35}`；packet size 與全幀 Laplacian 的相關係數：`0.5929`。
- 路牌徑向頻譜 cutoff 估計範圍：`0.3545`–`0.4077` cycles/pixel；這是操作性 threshold，不是光學 MTF。
- 路牌 ESF（erf fit、至少 30 條 profile）估計：垂直 sigma 中位數 `4.2150` px、10–90 寬度 `10.8036` px；水平 sigma `0.2754` px、寬度 `0.7060` px；水泥柱 frame 385 對照見 JSON。
- frame 385 水泥柱背景 ESF：`unavailable` sigma、`unavailable` px 10–90；若 invalid 已從統計排除。
- 中文大字高度估計中位數 `34.5000` px；估計筆畫寬度中位數 `5.0000` px。
- OCR：engine=`rapidocr-onnxruntime`, available=`True`；逐項命中統計見下方與 `diagnosis.json`。

**初步判定：** 原始影片是否形成資訊上限目前仍**未證實**；運動/失焦與低有效頻寬保留為中等候選，並等待軸向 edge spread 確認。這份診斷沒有產生新的增強、銳化、超解析或 final pipeline 輸出。

## 診斷 1：來源與壓縮

ffprobe 原始輸出摘要：

- codec/profile/pix_fmt：`h264` / `High` / `yuv420p`
- frame 350–385 I/P/B 分佈：`{'B': 1, 'P': 35}`
- bitrate 與每幀 packet size：完整資料在 `diagnosis.json` 的 `compression.frames_350_385`
- bitrate 只可判斷編碼負擔，不能單獨證明文字細節是否不存在；需配合頻譜與 edge spread。
- packet size 範圍與每幀 I/P/B、Laplacian 關聯均保留在 `diagnosis.json.compression`；相關係數不是因果證據。

## 診斷 2：有效解析度

frame 385 整張原始畫面做 1/2、1/3、1/4、1/6 降採樣再 bicubic 回放；PSNR/SSIM 與相鄰像素週期統計在 `diagnosis.json.effective_resolution`。這是「可被降採樣而近似保留」的操作性測試，不等於真實感測器解析度。

## 診斷 3：頻譜截止

![Radial spectra](radial_spectra.png)

每個 frame 350、360、370、375、380、385 同時量路牌與水泥柱/背景對照；使用去均值 + Hann + 2D FFT + 徑向平均功率。路牌 cutoff 若低於背景且字筆畫需要更高頻，支持「路牌本身資訊不足」；但 threshold 與 ROI 邊界會影響絕對值，故證據強度標為中至強，而非精密光學結論。

## 診斷 4：edge spread

固定使用三組垂直邊（左右外框、中間分隔線；邊緣沿 y、沿 x 量 sigma）與兩組水平邊（上下外框；邊緣沿 x、沿 y 量 sigma），另保留水泥柱背景。每個 ESF 沿邊緣平均至少 30 條原始灰階 profile；沒有插值、二值化或銳化。fit fail 或窗口邊界無效會標 invalid 並排除統計；逐幀 sigma/10–90、取樣線與軸定義在 `diagnosis.json`，診斷 overlay 為 `edge_spread_overlay_frame_385.png`。

## 診斷 5：運動模糊

Farneback flow 在原始 full-frame 上計算，再取當幀路牌 bbox 內位移；同時記錄 bbox 尺度變化。相鄰幀位移分開報告 median=`6.8644–15.5046` px/frame、mean=`6.0104–12.5376` px/frame。拖尾公式為 `flow_px_per_frame × 30 × exposure_seconds`，並分別列 mean/median；曝光仍未知，故只是情境估計，不能混用兩種統計口徑。

## 診斷 6：字元尺寸

以原始 crop 的亮低飽和連通元件估計大字、小字與筆畫尺寸。這不是 OCR ground truth，也可能受字元相連、壓縮與閾值影響；完整逐幀表在 `diagnosis.json.text_size`。一般 OCR 的 20–30 px 字高只是經驗參考，不是本片的保證門檻。

## 診斷 7：多幀融合資訊

本任務沒有新建 fusion/enhancement 圖。使用原始路牌 crop 以 frame 385 尺寸作共同座標的 leave-one-out proxy，並計算前後半序列穩定性；這會混入 resize/尺度變化，因此不是超解析證明。結果摘要：

- leave-one-out 單候選平均 PSNR=`17.0808` dB、SSIM=`0.5032`；
- 前後半序列彼此 PSNR=`18.7077` dB、SSIM=`0.6131`。

因此目前只支持「幀間內容差異很大、融合可作穩定化/降噪但有尺度與插值混淆」，不支持宣稱增加真實解析度。

## 診斷 8：OCR 基準

暫定真值為：`八德、鶯歌、Bade、Yingge、出口、EXIT、1½、公里、km、110乙`。使用 `rapidocr-onnxruntime`，逐次以隔離 worker 執行；控制組包含 36 個 raw crop、raw crop 的 2×/3× bicubic（僅 OCR 輸入，不保存視覺增強 final）、canonical 單幀（不融合）、canonical median（不做 CLAHE）與既有完整 pipeline 輸出。每個輸入記錄辨識文字、confidence、10 個真值逐項命中，以及大字/小字命中。尺寸/矩形化、融合與 CLAHE 的控制定義寫在 `diagnosis.json.ocr.inputs`。摘要：`{"raw_frames_350_385": {"runs": 36, "available_runs": 36, "mean_hit_count": 1.6111111111111112, "big_hit_count": 0.6666666666666666, "small_hit_count": 0.9444444444444444}, "raw_frames_2x_bicubic": {"runs": 36, "available_runs": 36, "mean_hit_count": 1.6111111111111112, "big_hit_count": 0.4444444444444444, "small_hit_count": 1.1666666666666667}, "raw_frames_3x_bicubic": {"runs": 36, "available_runs": 36, "mean_hit_count": 1.5833333333333333, "big_hit_count": 0.4722222222222222, "small_hit_count": 1.1111111111111112}, "canonical_single_no_fusion": {"runs": 1, "available_runs": 1, "mean_hit_count": 4.0, "big_hit_count": 1.0, "small_hit_count": 3.0}, "canonical_median_no_clahe": {"runs": 1, "available_runs": 1, "mean_hit_count": 3.0, "big_hit_count": 1.0, "small_hit_count": 2.0}, "full_pipeline_contrast_after_fusion": {"runs": 1, "available_runs": 1, "mean_hit_count": 4.0, "big_hit_count": 1.0, "small_hit_count": 3.0}}`。

逐項命中（至少一筆可用輸入命中；`1` 不是字串正確率）：

| 控制組 | 八德 | 鶯歌 | Bade | Yingge | 出口 | EXIT | 1½ | 公里 | km | 110乙 |
|---|---|---|---|---|---|---|---|---|---|---|
| raw_frames_350_385 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| raw_frames_2x_bicubic | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| raw_frames_3x_bicubic | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| canonical_single_no_fusion | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| canonical_median_no_clahe | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| full_pipeline_contrast_after_fusion | 1 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |

大字/小字分組平均命中數（大字 `八德, 鶯歌`；小字 `出口, 公里, EXIT, km, Bade, Yingge, 1½, 110乙`）：`{"raw_frames_350_385": {"big_mean": 0.6666666666666666, "small_mean": 0.9444444444444444}, "raw_frames_2x_bicubic": {"big_mean": 0.4444444444444444, "small_mean": 1.1666666666666667}, "raw_frames_3x_bicubic": {"big_mean": 0.4722222222222222, "small_mean": 1.1111111111111112}, "canonical_single_no_fusion": {"big_mean": 1.0, "small_mean": 3.0}, "canonical_median_no_clahe": {"big_mean": 1.0, "small_mean": 2.0}, "full_pipeline_contrast_after_fusion": {"big_mean": 1.0, "small_mean": 3.0}}`。

## 判定表

| 問題 | 結論 | 證據強度 | 依據 |
|---|---|---|---|
| 資訊上限是否由原始影片決定？ | 未證實 | 未證實 | 頻譜、筆畫尺寸與降採樣只是操作性證據；尚缺軸向 edge spread 確認與可歸因的保留幀驗證 |
| 主要模糊來源 | 運動/失焦/低有效細節仍是候選 | 中（待軸向 edge spread 確認） | flow 與方向性 edge spread 提供候選訊號，但不能把異向性直接歸因 |
| 多幀融合是否增加真實解析度？ | 不支持宣稱；較安全解釋是穩定化/降噪 | 中 | raw-crop leave-one-out proxy 受尺度/插值混淆，未得到可歸因於真實高頻新增的證據 |
| 是否有任何版本可被 OCR 讀出？ | `True` | 中（若 worker 成功）/弱（若 unavailable） | RapidOCR 實際輸出與逐項命中表；輸入尺寸、矩形化、融合、CLAHE 混淆已分組標註 |

## 下一步（限定選項）

- **B. 改取更好的輸入（首選）**：更高碼率、更近/更早幀、另一支影片，最直接針對已觀察的原始資訊上限。
- **A. 繼續改善演算法（次選）**：只在新輸入顯示可挖掘高頻資訊時進行；先做 raw-frame OCR/CER 與保留幀驗證。
- **C. 改目標為文字辨識**：若業務只需固定路牌內容，使用已知字型/候選字串模板比對，承認這不是像素恢復。
- **D. 學習式模型**：最後選項；輸出必須標註視覺增強並以 OCR/保留幀一致性驗證，不能當真實字形證據。

## 假設與未知

1. 真值字串依使用者指定暫定，尚未取得人工確認。
2. 曝光時間未知；拖尾只列三種情境。
3. HSV bbox 是量測用 ROI，不是通用路牌 detector。
4. edge spread 未經鏡頭標定；目前方向性差異仍可能受 bbox/中線座標影響，須以軸向邊緣確認。
5. OCR 模型可由執行環境提供或下載權重；實際可用狀態與 worker 原因以 `diagnosis.json` 為準。

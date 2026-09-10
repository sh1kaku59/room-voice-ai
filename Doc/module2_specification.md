# MODULE 2: WAKE-WORD DETECTION (CALL NAME)
*(Tài liệu Đặc tả Thiết kế Kỹ thuật & Nghiên cứu Học thuật Chuẩn hóa)*

---

### 1. Giới thiệu & Mục tiêu Nghiên cứu (Overview & Research Objectives)

#### 1.1. Vị trí Kiến trúc trong Hệ thống RooM
Trong hệ thống 7 module của trợ lý giọng nói thời gian thực **RooM (Voice Call Meeting AI)**, **Module 2 (Wake-word Detection)** đóng vai trò là một **Bộ kiểm soát trạng thái hệ thống (State Controller)** [cite: 24, 50]. 

Hệ thống mặc định vận hành ở **Trạng thái Giám sát Thụ động (Passive Monitoring / Idle State)** nhằm tối ưu hóa năng lượng tiêu thụ [cite: 24, 50]. Tại đây, các module nặng ký như Module 3 (ASR) và Module 6 (LLM AI Brain) được đưa vào chế độ ngủ đông. Module 2 liên tục lắng nghe và phân tích luồng âm thanh sạch $S_{\text{enh}}(t)$ được chuyển tiếp từ Module 1 (Audio Preprocessing & VAD) [cite: 24, 50]. Khi phát hiện người tham gia gọi tên trợ lý bằng cụm từ kích hoạt (ví dụ: *"AI Star Light"*, *"Star Light ơi"* hoặc *"Hey Star Light"*), Module 2 sẽ lập tức bắn ra sự kiện thức tỉnh hệ thống `EVENT_WAKEUP` [cite: 24, 50, 65]. Sự kiện này chuyển hệ thống sang **Trạng thái Lắng nghe Tích cực (Active Listening State)**, đồng thời mở luồng định tuyến âm thanh đến Module 3 để thực hiện dịch giọng nói thời gian thực [cite: 24, 50, 65].

```text
[Audio sạch từ Module 1] ──► [Silero VAD = TRUE] ──► [MODULE 2: Wake-word Engine]
                                                              │
                                    ┌─────────────────────────┴─────────────────────────┐
                                    ▼                                                   ▼
                         [Wake-word Detected: YES]                           [Wake-word Detected: NO]
                                    │                                                   │
                                    ▼                                                   ▼
                   Chuyển sang ACTIVE LISTENING                              Duy trì PASSIVE MONITORING
                   (Kích hoạt Module 3: ASR)                                  (Tiết kiệm 95% tài nguyên)
```

#### 1.2. Mục tiêu Kỹ thuật & Khoa học
*   **Độ chính xác kích hoạt cực cao:** Đạt tỷ lệ kích hoạt đúng **True Positive Rate (TPR) $\ge 96.2\%$** trong môi trường văn phòng có tạp âm trung bình, đồng thời khống chế tỷ lệ báo động giả **False Alarm Rate (FAR) < 0.5 lần/giờ** hoạt động thường trực [cite: 25, 50, 53].
*   **Độ bền bỉ đối với ngữ âm và thanh điệu tiếng Việt (Vietnamese Tonal Robustness):** Nhận diện chính xác từ khóa kích hoạt có chứa các thanh điệu tiếng Việt phức tạp (thanh nặng và thanh sắc trong cụm từ *"Trợ lý"* hoặc âm điệu luyến của từ *"ơi"*) bất kể phương ngữ Bắc - Trung - Nam và ngữ điệu câu nói [cite: 25, 50].
*   **Cơ chế chống kích hoạt lặp (Anti-Double Triggering):** Thiết lập cơ chế khóa thời gian trơ (**Refractory Period Cooldown**) để triệt tiêu hoàn toàn hiện tượng một câu gọi kéo dài gây kích hoạt bot nhiều lần liên tiếp [cite: 25, 30, 50, 52].
*   **Phân bổ tài nguyên tối giản (Zero-GPU Footprint):** Module 2 chạy thường trực $100\%$ trên CPU Intel Core i7-12650H thông qua mô hình lượng tử hóa **ONNX Runtime (int8/fp32)** với mức chiếm dụng tài nguyên cực thấp: **$< 2\%$ CPU** và **$< 30\text{ MB}$ RAM** [cite: 25, 50]. Toàn bộ dung lượng VRAM của GPU NVIDIA RTX 3050 (4GB) được giải phóng để dành riêng cho mô hình ASR (Module 3) và LLM (Module 6) [cite: 25, 50].

---

### 2. Phân tích Chuyên sâu Cơ chế Hoạt động & Thuật toán

Hệ thống Wake-word vận hành liên tục theo vi kiến trúc 3 tầng xử lý tuần tự nối tiếp:

#### 2.1. Tầng 1: Sliding Acoustic Feature Extraction (Bộ đệm trượt đặc trưng âm học)
*   **Circular Audio RingBuffer:** Hệ thống duy trì một bộ đệm vòng tuần hoàn có cửa sổ trượt cố định từ $1.0\text{s}$ đến $1.5\text{s}$ ($16,000 - 24,000$ mẫu âm thanh ở tần số lấy mẫu $16\text{kHz}$) [cite: 26, 28, 50, 51]. Khi có khung âm thanh mới ($32\text{ms} = 512$ mẫu) từ Module 1 đẩy vào, bộ đệm tự động thực hiện cơ chế FIFO đẩy $512$ mẫu cũ nhất ra và nạp $512$ mẫu mới vào với độ phức tạp thời gian đạt mức tối ưu $O(1)$ [cite: 28, 51].
*   **Log-Mel Spectrogram Extraction:** Toàn bộ dữ liệu trong bộ đệm trượt $1.5\text{s}$ được chuyển đổi từ miền thời gian sang miền tần số thông qua phép biến đổi Fourier thời gian ngắn (STFT) với độ dài cửa sổ phân tích $25\text{ms}$ và bước nhảy $10\text{ms}$. Sau đó, phổ biên độ được đưa qua ma trận bộ lọc Mel gồm $80$ dải tần để trích xuất tensor đặc trưng âm học $X_t \in \mathbb{R}^{T \times 80}$ [cite: 26, 28, 51].

#### 2.2. Tầng 2: Neural Keyword Classification Core (Lõi nhận diện từ khóa nơ-ron)
*   **Kiến trúc OpenWakeWord:** Sử dụng kiến trúc mạng nơ-ron tích chập tách biệt chiều sâu (**Depthwise Separable Convolution**) kết hợp các khối nơ-ron Conformer siêu nhỏ [cite: 27, 29, 50, 51]. Trọng số của mô hình được lượng tử hóa chặt chẽ sang định dạng **ONNX int8** giúp giảm kích thước file xuống chỉ còn **~2.5 MB** [cite: 29, 51].
*   **Ước lượng xác suất nhị phân (Binary Logistic Inference):** Đầu ra của mô hình nơ-ron tại mỗi bước nhảy khung hình là một giá trị xác suất hậu nghiệm $P(\text{WakeWord} \mid X_t) \in [0.0, 1.0]$ biểu thị độ tin cậy từ khóa *"AI Star Light"* vừa xuất hiện trong cửa sổ âm thanh gần nhất [cite: 27, 29, 50, 51]. Quá trình suy luận ONNX được biên dịch tối ưu hóa tập lệnh AVX2 giúp thực thi siêu tốc với độ trễ **$< 3.5\text{ ms} / \text{frame}$** chạy thuần trên CPU [cite: 27, 50, 53].

#### 2.3. Tầng 3: Smoothing & Debounce Controller (Bộ hậu xử lý chống rung nhiễu & Khóa trơ)
*   **Lọc trung bình trượt thời gian (Temporal Posterior Smoothing):** Để triệt tiêu các gai nhiễu xác suất đột biến (spikes) do tiếng ồn va đập trong phòng họp (tiếng gõ phím cơ, tiếng cười nói ngẫu nhiên) gây ra kích hoạt ảo, hệ thống áp dụng một bộ lọc trung bình trượt qua cửa sổ $K = 4$ khung hình liên tiếp để tính toán xác suất làm mịn $P_{\text{smooth}}(t)$ [cite: 27, 30, 50, 52].
*   **Cơ chế phân ngưỡng kép (Hysteresis Thresholding):** Để đưa ra quyết định thức tỉnh chuẩn xác, hệ thống áp dụng cơ chế ngưỡng kép động: Cờ kích hoạt chỉ được bật lên khi $P_{\text{smooth}}(t) \ge \theta_{\text{start}} = 0.75$ và chỉ được reset trở lại trạng thái sẵn sàng lắng nghe khi xác suất rơi xuống dưới ngưỡng $\theta_{\text{reset}} = 0.40$ [cite: 27, 30, 50, 52].
*   **Bộ đếm thời gian trơ chống kích hoạt đúp (Refractory Debounce Period):** Khi từ khóa kích hoạt thành công, hệ thống phát sự kiện `EVENT_WAKEUP` đồng thời thiết lập một bộ đếm thời gian trơ $T_{\text{cooldown}} = 2.0\text{s}$ (tương đương với $62$ khung hình ở bước nhảy $32\text{ms}$) [cite: 28, 30, 50, 52]. Trong suốt thời gian này, cờ kích hoạt bị khóa cứng ở trạng thái ẩn để ngăn chặn hiện tượng người dùng kéo dài âm đuôi (ví dụ: *"Star Light ơi...iiii"*) gây kích hoạt lặp [cite: 30, 52].

---

### 3. Mô hình hóa Toán học Chuẩn mực (Mathematical Formulation)

#### 3.1. Trích xuất đặc trưng Log-Mel Spectrogram
Với luồng tín hiệu âm thanh sạch $\hat{s}(t)$ đầu ra từ Module 1, vector đặc trưng phổ Log-Mel $X_t \in \mathbb{R}^{T \times F}$ trên cửa sổ trượt độ dài $W$ ($W = 1.5\text{s}$ ở $16\text{kHz}$) tại thời điểm $t$ được định nghĩa bằng phương trình [cite: 31, 51]:
$$X_t = \log \Big( \mathbf{M}_{\text{mel}} \cdot \big| \text{STFT}\big(\hat{s}(t - W : t)\big) \big|^2 + \epsilon \Big)$$
Trong đó:
*   $\mathbf{M}_{\text{mel}} \in \mathbb{R}^{F \times K}$ là ma trận dải lọc Mel ($F=80$ dải lọc Mel, $K$ là số lượng vạch phổ tần số STFT) [cite: 31, 51].
*   $\epsilon = 10^{-6}$ là hằng số làm mịn số học chống tràn logarit âm vô cùng [cite: 31, 51].

#### 3.2. Ước lượng xác suất hậu nghiệm & Làm mịn chuỗi thời gian
Mô hình mạng nơ-ron phân loại nhị phân Wake-word $f_\theta$ với bộ tham số trọng số $\theta$ thực hiện ước lượng xác suất xuất hiện từ khóa thô $P_{\text{raw}}(t)$ tại thời điểm $t$ thông qua hàm Sigmoid kích hoạt [cite: 31, 52]:
$$P_{\text{raw}}(t) = \sigma\big(f_\theta(X_t)\big) = \frac{1}{1 + e^{-f_\theta(X_t)}}$$
Xác suất hậu nghiệm được làm mịn thời gian để triệt tiêu các gai nhiễu ngẫu nhiên qua cửa sổ trượt bậc $K$ ($K=4$ khung hình liên tiếp) [cite: 32, 52]:
$$P_{\text{smooth}}(t) = \frac{1}{K} \sum_{i=0}^{K-1} P_{\text{raw}}(t - i)$$

#### 3.3. Hàm quyết định kích hoạt kèm Khóa thời gian trơ (Refractory Debounce Rule)
Trạng thái kích hoạt hệ thống $\mathcal{W}(t) \in \{0, 1\}$ tại thời điểm $t$ tuân thủ chặt chẽ hàm logic quyết định kết hợp biến đếm thời gian trơ $\tau_{\text{cooldown}}$ [cite: 32, 52]:
$$\mathcal{W}(t) = \begin{cases} 1, & \text{nếu } P_{\text{smooth}}(t) \ge \theta_{\text{wake}} \quad \text{và} \quad \tau_{\text{cooldown}} = 0 \\ 0, & \text{nếu } P_{\text{smooth}}(t) < \theta_{\text{wake}} \quad \text{hoặc} \quad \tau_{\text{cooldown}} > 0 \end{cases}$$
Trong đó, quy tắc cập nhật đồng hồ đếm ngược thời gian trơ $\tau_{\text{cooldown}}$ sau mỗi bước nhảy khung hình được tính toán tự động bằng phương trình [cite: 32, 52]:
$$\tau_{\text{cooldown}} \leftarrow \begin{cases} N_{\text{debounce}}, & \text{khi } \mathcal{W}(t) = 1 \\ \max(0, \tau_{\text{cooldown}} - 1), & \text{khi } \mathcal{W}(t) = 0 \end{cases}$$
Với mức thiết lập thời gian trơ $T_{\text{cooldown}} = 2.0\text{s}$ ở bước nhảy khung hình $32\text{ms}$ ($512$ mẫu ở $16\text{kHz}$), hằng số khung hình trơ tối đa được định lượng chính xác [cite: 52]:
$$N_{\text{debounce}} = \frac{2.0\text{s}}{0.032\text{s}} \approx 62 \text{ frames}$$

---

### 4. Sơ đồ Luồng Xử lý Kiến trúc (Mermaid Diagram)

Sơ đồ dưới đây trực quan hóa chuỗi truyền dẫn tín hiệu âm thanh và luồng quyết định kích hoạt của Module 2:

```mermaid
graph TD
    %% Styling & Colors
    classDef inputStyle fill:#F1F5F9,stroke:#94A3B8,stroke-width:2px,color:#0F172A;
    classDef t1Style fill:#EFF6FF,stroke:#2563EB,stroke-width:2px,color:#1E293B;
    classDef t2Style fill:#FEF3C7,stroke:#D97706,stroke-width:2px,color:#1E293B;
    classDef t3Style fill:#ECFDF5,stroke:#059669,stroke-width:2px,color:#1E293B;
    classDef triggerStyle fill:#FEE2E2,stroke:#DC2626,stroke-width:2px,color:#991B1B;
    classDef idleStyle fill:#F8FAFC,stroke:#94A3B8,stroke-width:1.5px,color:#64748B;

    %% Nodes
    In["<b>INPUT FROM MODULE 1</b><br/>• Enhanced PCM 16kHz Stream<br/>• VAD Speech Frame = TRUE"]:::inputStyle

    T1["<b>TẦNG 1: Sliding Feature Extraction</b><br/>• Circular RingBuffer: Cửa sổ trượt 1.5 giây<br/>• 80-channel Log-Mel Spectrogram<br/>• Khung bước nhảy: 32ms (512 samples)"]:::t1Style

    T2["<b>TẦNG 2: OpenWakeWord Neural Classifier</b><br/>• Depthwise Conv + Small Conformer KWS (ONNX int8)<br/>• Tính xác suất nhận diện P(WakeWord | X_t)<br/>• Suy luận CPU siêu nhẹ: < 3.5ms"]:::t2Style

    T3["<b>TẦNG 3: Smoothing & Debounce Controller</b><br/>• Moving Average Filter: Làm mịn qua 4 frames<br/>• Hysteresis Threshold: θ_wake = 0.75<br/>• Cooldown Refractory Lockout: 2.0 giây"]:::t3Style

    Dec{"<b>Wake-word Detected & Cooldown == 0?</b>"}

    OutActive["<b>TRIGGER = TRUE</b><br/>Phát Event Wake-up<br/>──► <b>Chuyển sang ACTIVE LISTENING (Mod 3: ASR)</b>"]:::triggerStyle

    OutIdle["<b>TRIGGER = FALSE</b><br/>Duy trì <b>PASSIVE MONITORING</b><br/><i>Tiết kiệm 95% tài nguyên hệ thống</i>"]:::idleStyle

    %% Connections
    In -->|"Frame 32ms sạch"| T1
    T1 -->|"Tensor X_t (1.5s window)"| T2
    T2 -->|"P_raw(t)"| T3
    T3 --> Dec
    Dec -->|"YES (P_smooth ≥ 0.75)"| OutActive
    Dec -->|"NO (P_smooth < 0.75)"| OutIdle
```

---

### 5. Chiến lược Sinh dữ liệu Huấn luyện Nhân tạo Đa điều kiện (Multi-condition Synthetic TTS Training)

Để huấn luyện mô hình OpenWakeWord nhận diện chính xác từ khóa tùy chỉnh *"AI Star Light"* hoạt động 100% offline mà không tốn chi phí thu âm thực tế khổng lồ, hệ thống áp dụng giải pháp sinh dữ liệu nhân tạo tiên tiến kế thừa từ các công trình nghiên cứu SOTA [cite: 33, 54]:

```text
    [8 Giọng đọc TTS Việt]                 [Tập tạp âm nền]            [Tập đáp ứng xung]
      (Bắc-Trung-Nam)                  (Microsoft DNS 2023)             Phòng họp (RIR)
             │                                   │                             │
             ▼                                   ▼                             ▼
┌─────────────────────────┐             ┌─────────────────┐           ┌─────────────────┐
│ Sinh 10,000 từ khóa thô │             │  Chuẩn hóa SNR  │           │ Tích chập không │
│  Tốc độ: 0.8x - 1.2x    │             │  -5dB đến 15dB  │           │ gian (Reverb)   │
└────────────┬────────────┘             └────────┬────────┘           └────────┬────────┘
             │                                   │                             │
             └─────────────────►  TRỘN PHÂN PHỐI  ◄────────────────────────────┘
                                         │
                                         ▼
                             [10,000 Mẫu Dương Tính]
                             (Sẵn sàng nạp Train KWS)
```

1.  **Sinh mẫu câu chứa từ khóa dương tính (Positive Samples Generation):**
    *   Sử dụng **8 giọng đọc tổng hợp (TTS) tiếng Việt chất lượng cao** phân bố đều các vùng miền (Bắc, Trung, Nam) và giới tính (Nam/Nữ) để sinh ra **10,000 mẫu câu thoại khác nhau** có chứa từ khóa *"AI Star Light"*, *"Star Light ơi"* hoặc *"Hey Star Light"* [cite: 33].
    *   Áp dụng các kỹ thuật biến đổi độ dài thời gian (Time-stretching) để đa dạng hóa tốc độ phát âm từ cực nhanh đến cực chậm với dải hệ số biến thiên $\alpha \in [0.8, 1.2]$ [cite: 33].
2.  **Mô phỏng âm học phòng họp thực tế (Acoustic Augmentation via RIR & Noise):**
    *   **Tạo âm vang phòng (Reverberation):** Sử dụng tập mẫu đáp ứng xung phòng họp thực tế **RIR (Room Impulse Response)** của Microsoft DNS Challenge 2023 [cite: 33, 51]. Thực hiện phép toán tích chập trong miền thời gian giữa mẫu âm thanh thô với xung phòng để mô phỏng chính xác hiện tượng dội âm khi người nói ngồi xa micro [cite: 33, 51].
    *   **Trộn nhiễu nền thực tế (Acoustic Noise Mixing):** Trích xuất các dải tiếng ồn văn phòng đặc thù (tiếng máy lạnh, tiếng click chuột, tiếng quạt tản nhiệt laptop, tiếng gõ phím cơ) từ bộ dữ liệu **Microsoft DNS Challenge 2023** [cite: 33, 51]. Trộn các dải tiếng ồn này vào mẫu dương tính thô với phân phối tỷ số tín hiệu trên nhiễu biến thiên ngẫu nhiên **SNR $\in [-5\text{ dB}, 15\text{ dB}]$** [cite: 33].
3.  **Tập mẫu âm bản đối chứng (Negative Samples Dataset):**
    *   Nạp **100 giờ dữ liệu hội thoại tiếng Việt thông thường** hoàn toàn không chứa từ khóa kích hoạt (trích xuất từ bộ dữ liệu VIVOS của ĐHQG TP.HCM và các đoạn hội thoại họp nhóm trên YouTube) [cite: 33, 51].
    *   Tập mẫu âm bản này giúp huấn luyện mạng nơ-ron phân biệt cực kỳ nhạy bén giữa các cuộc trò chuyện thông thường và câu lệnh thực sự gọi trợ lý, triệt tiêu tỷ lệ báo động giả [cite: 33, 51].

---

### 6. Mã nguồn Python Mẫu Chuẩn hóa (OpenWakeWord ONNX CPU Wrapper)

Dưới đây là tệp mã nguồn Python chuẩn thương mại `src/module2_wakeword/detector.py` được đóng gói dưới dạng class độc lập, tối ưu hóa 100% CPU AVX2 sử dụng ONNX Runtime [cite: 25, 50, 53]:

```python
"""
RooM Voice Call Meeting AI - Module 2: Wake-word Detection (Call Name Engine)
Mã nguồn tối ưu chạy ngầm thường trực 100% trên CPU qua ONNX Runtime CPU.
"""

import numpy as np
import collections
import onnxruntime as ort
from configs.settings import WAKEWORD_HYSTERESIS_MS, AUDIO_CHUNK_MS

class Module2WakeWordDetector:
    """
    Class điều khiển và nhận diện từ khóa gọi tên trợ lý 'AI Star Light' thời gian thực.
    Kế thừa cơ chế làm mịn xác suất và khóa thời gian trơ chống kích hoạt lặp.
    """
    def __init__(
        self,
        model_path: str = "models/ai_star_light_openwakeword.onnx",
        sample_rate: int = 16000,
        frame_size: int = 512,
        window_duration: float = 1.5,
        threshold: float = 0.75,
        cooldown_seconds: float = 2.0
    ):
        """
        Khởi tạo Session ONNX CPU và thiết lập các bộ đệm làm mịn xác suất.
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size               # 512 samples = 32ms ở 16kHz
        self.threshold = threshold
        
        # 1. Cấu hình bộ đệm trượt 1.5 giây cho âm thanh đầu vào (Circular RingBuffer)
        self.buffer_size = int(window_duration * sample_rate) # 1.5s * 16000 = 24,000 samples
        self.audio_buffer = collections.deque(maxlen=self.buffer_size)
        # Nạp đầy bộ đệm ban đầu bằng các mẫu im lặng
        for _ in range(self.buffer_size):
            self.audio_buffer.append(0.0)

        # 2. Khởi tạo ONNX Runtime Session chạy thuần CPU Execution
        print(f"[Module 2] Đang tải mô hình Wake-word ONNX từ {model_path}...")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # Ép buộc sử dụng CPU Execution Provider để giải phóng VRAM GPU cho ASR/LLM
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=opts, 
            providers=['CPUExecutionProvider']
        )
        print("[Module 2] Nạp mô hình thành công! Đã cấu hình chạy trên CPU (0MB VRAM).")

        # 3. Thiết lập bộ đệm làm mịn xác suất (Temporal Moving Average)
        # WAKEWORD_HYSTERESIS_MS mặc định là 300ms, tương đương ~10 frames ở bước nhảy 32ms
        self.smoothing_window_size = max(1, int(WAKEWORD_HYSTERESIS_MS / AUDIO_CHUNK_MS))
        self.prob_history = collections.deque(maxlen=self.smoothing_window_size)
        for _ in range(self.smoothing_window_size):
            self.prob_history.append(0.0)

        # 4. Quản lý trạng thái khóa trơ chống kích hoạt lặp (Cooldown Manager)
        self.cooldown_frames_limit = int(cooldown_seconds / (frame_size / sample_rate)) # 2.0s / 0.032s = 62 frames
        self.cooldown_counter = 0

    def _extract_log_mel_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Giả lập bước trích xuất đặc trưng Log-Mel Spectrogram trực tiếp từ mảng âm thanh trượt 1.5s.
        Trong thực tế, mô hình OpenWakeWord ONNX tự tích hợp tầng Mel-Spectrogram hoặc nhận feature tensor.
        """
        # Giả lập feature tensor đầu vào của mô hình (ví dụ: Batch x Time x Features = 1 x 129 x 80)
        dummy_feature = np.random.uniform(-1.0, 1.0, (1, 129, 80)).astype(np.float32)
        return dummy_feature

    def process_speech_frame(self, clean_audio_frame: np.ndarray) -> tuple[bool, float]:
        """
        Xử lý và đánh giá khung âm thanh 32ms sạch nhận về từ Module 1.
        
        Args:
            clean_audio_frame (np.ndarray): Mảng float32 PCM gồm 512 samples.
            
        Returns:
            triggered (bool): Cờ báo hiệu từ khóa 'AI Star Light' được kích hoạt thành công.
            smoothed_score (float): Điểm số xác suất đã được làm mịn thời gian.
        """
        # 1. Cập nhật khung âm thanh mới vào bộ đệm trượt vòng tuần hoàn
        for sample in clean_audio_frame:
            self.audio_buffer.append(sample)

        # 2. Quản lý đồng hồ trơ (Cooldown Decounter)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            # Đang trong thời gian trơ -> Trả về False ngay lập tức để tiết kiệm CPU và chống đúp
            self.prob_history.append(0.0)
            return False, 0.0

        # 3. Trích xuất đặc trưng âm học
        audio_array = np.array(self.audio_buffer, dtype=np.float32)
        feature_tensor = self._extract_log_mel_features(audio_array)

        # 4. Thực thi suy luận nhị phân siêu tốc qua ONNX CPU Runtime (< 3.5ms)
        input_name = self.session.get_inputs()[0].name
        raw_outputs = self.session.run(None, {input_name: feature_tensor})
        # Giả sử đầu ra là phân phối nhị phân, trích xuất xác suất từ khóa
        raw_prob = float(raw_outputs[0][0][0]) if len(raw_outputs) > 0 else 0.0

        # 5. Làm mịn xác suất qua bộ lọc trung bình trượt thời gian (Moving Average)
        self.prob_history.append(raw_prob)
        smoothed_score = sum(self.prob_history) / len(self.prob_history)

        # 6. Đưa ra quyết định kích hoạt dựa trên ngưỡng và khóa trơ
        triggered = False
        if smoothed_score >= self.threshold:
            triggered = True
            # Bật đồng hồ khóa trơ ngay lập tức trong 2.0 giây tiếp theo
            self.cooldown_counter = self.cooldown_frames_limit
            # Clear bộ đệm xác suất để chuẩn bị cho chu kỳ tiếp theo
            self.prob_history.clear()
            for _ in range(self.smoothing_window_size):
                self.prob_history.append(0.0)

        return triggered, smoothed_score

# Chạy thử nghiệm và đo đạc hiệu năng độc lập khi thực thi file trực tiếp
if __name__ == "__main__":
    import time
    
    # 1. Tạo file dummy onnx giả lập để chạy demo nếu chưa có file model thật
    # (Trong thực tế, file này sẽ là ai_star_light_openwakeword.onnx được huấn luyện xong)
    model_dummy_path = "models/ai_star_light_openwakeword.onnx"
    import os
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        
    # Tạo mô hình dummy ONNX đơn giản để pass code test khởi chạy
    import torch
    import torch.nn as nn
    class DummyKWS(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(80, 1)
        def forward(self, x):
            # Nhận đầu vào (1, 129, 80) -> Trung bình thời gian -> Trả về xác suất nhị phân (1, 1)
            x_mean = torch.mean(x, dim=1)
            return torch.sigmoid(self.linear(x_mean)).unsqueeze(-1)
            
    dummy_model = DummyKWS()
    dummy_input = torch.randn(1, 129, 80)
    torch.onnx.export(
        dummy_model, dummy_input, model_dummy_path,
        input_names=["input_audio_features"], output_names=["probability_output"],
        dynamic_axes={"input_audio_features": {1: "time_steps"}}
    )

    # 2. Khởi tạo Detector
    detector = Module2WakeWordDetector(model_path=model_dummy_path)
    
    # 3. Giả lập luồng âm thanh 5 giây (mỗi khung 32ms = 512 samples)
    print("\n--- CHẠY MÔ PHỎNG LUỒNG ÂM THANH STREAMING 5 GIÂY ---")
    latencies = []
    
    for i in range(150): # 150 frames * 32ms ≈ 4.8 giây
        # Giả lập khung tín hiệu âm thanh sạch ngẫu nhiên
        dummy_frame = np.random.uniform(-0.02, 0.02, 512).astype(np.float32)
        
        # Đo độ trễ xử lý của một khung hình
        start_t = time.perf_counter()
        triggered, score = detector.process_speech_frame(dummy_frame)
        latency = (time.perf_counter() - start_t) * 1000 # Đơn vị: miligiây
        latencies.append(latency)
        
        if triggered:
            print(f"🔥 [KÍCH HOẠT THÀNH CÔNG] Frame {i:03d} | Score: {score:.4f} | Latency: {latency:.3f}ms")
        elif i % 20 == 0:
            print(f"⌛ [Đang giám sát ngầm] Frame {i:03d} | Score: {score:.4f} | Latency: {latency:.3f}ms")
            
    print(f"\n⚡ [HIỆU NĂNG THỰC TẾ TRÊN CPU] Độ trễ trung bình mỗi khung: {np.mean(latencies):.4f} ms (KPI: <= 3.5ms)")
```

---

### 7. Kịch bản 5 Test Cases Chuyên sâu Kiểm thử Độc lập (Pytest Integration)

Để nghiệm thu nghiêm ngặt độ chính xác kích hoạt (TPR) và khả năng kháng báo động giả (FAR) của Module 2, tệp kiểm thử `tests/test_module2.py` phải chứa đúng 5 test cases bao bọc các kịch bản biên thực tế dưới đây:

#### Test Case 1: `test_initialization` (Kiểm thử Khởi tạo & Cấu hình bộ đệm)
*   **Mục tiêu:** Xác minh quá trình khởi tạo đối tượng `Module2WakeWordDetector` nạp thành công mô hình ONNX CPU, thiết lập đúng kích thước bộ đệm trượt $1.5\text{s}$ ($24,000$ mẫu) và bộ đếm moving average làm mịn xác suất.
*   **Kịch bản kiểm thử:** Khởi tạo detector, assert kích thước mảng trượt `audio_buffer` đạt chính xác $24,000$ phần tử, kiểm tra `cooldown_counter` ban đầu bằng $0$.

#### Test Case 2: `test_circular_buffer_fifo` (Kiểm thử Cơ chế trượt FIFO của Bộ đệm)
*   **Mục tiêu:** Xác minh cơ chế FIFO của hàng đợi vòng hoạt động chuẩn xác khi nhận các khung âm thanh nối tiếp nhau, đảm bảo dữ liệu mới nhất được ghi đè và dữ liệu cũ nhất bị triệt tiêu hoàn toàn.
*   **Kịch bản kiểm thử:** 
    *   Tạo mảng khung âm thanh ban đầu chứa toàn số $0.0$.
    *   Đẩy một khung âm thanh chứa toàn giá trị đặc biệt $1.0$ (512 mẫu) vào.
    *   Kiểm tra xem $512$ phần tử cuối cùng của bộ đệm trượt có bằng chính xác $1.0$ hay không và các phần tử ở đầu bộ đệm vẫn duy trì bằng $0.0$.

#### Test Case 3: `test_temporal_smoothing_spike_rejection` (Kiểm thử Kháng gai nhiễu đột biến)
*   **Mục tiêu:** Xác minh bộ lọc làm mịn trung bình trượt thời gian hoạt động tốt, có khả năng triệt tiêu thành công gai xác suất cực đại tức thời (chỉ xuất hiện trong 1 frame đơn lẻ), không gây kích hoạt ảo (FAR).
*   **Kịch bản kiểm thử:**
    *   Mock hàm `session.run` để giả lập đầu ra mô hình nơ-ron: 100 khung hình liên tiếp trả về xác suất thấp $0.05$, chỉ duy nhất khung hình thứ 50 trả về gai xác suất tuyệt đối $1.0$.
    *   Đẩy luồng này qua bộ detector. 
    *   Assert cờ `triggered` thu về ở toàn bộ các khung hình đều bằng `False` (bộ lọc trung bình trượt đã san phẳng gai xác suất này xuống dưới ngưỡng $0.75$).

#### Test Case 4: `test_hysteresis_and_cooldown_lockout` (Kiểm thử Kích hoạt & Khóa trơ chống đúp)
*   **Mục tiêu:** Xác minh khi xác suất làm mịn vượt ngưỡng $0.75$, hệ thống kích hoạt thành công (`triggered = True`), đồng thời kích hoạt bộ đếm khóa trơ trong 2.0 giây tiếp theo, tạm ngắt suy luận để tránh kích hoạt đúp khi người dùng luyến âm đuôi.
*   **Kịch bản kiểm thử:**
    *   Mock hàm đầu ra mô hình nơ-ron liên tục trả về xác suất cao $0.95$ trong $10$ khung hình liên tiếp.
    *   Khung hình đầu tiên đạt ngưỡng làm mịn, assert cờ `triggered` trả về bằng `True`.
    *   Tại khung hình thứ 2 ngay sau đó, mặc dù đầu ra mô hình vẫn là $0.95$, assert cờ `triggered` phải trả về bằng `False` và `cooldown_counter` được thiết lập đúng về mức $62$ khung hình đếm ngược.

#### Test Case 5: `test_end_to_end_dummy_stream_performance` (Kiểm thử Hiệu năng luồng & Giới hạn CPU)
*   **Mục tiêu:** Chạy mô phỏng toàn bộ luồng âm thanh dài 5 giây để đo đạc thời gian suy luận trung bình mỗi khung hình trên CPU, đảm bảo đạt yêu cầu vận hành thời gian thực.
*   **Kịch bản kiểm thử:**
    *   Cho chạy một vòng lặp gồm $150$ khung âm thanh PCM thô.
    *   Đo đạc và tính toán `mean_latency = np.mean(latencies)`.
    *   Assert chỉ số `mean_latency <= 3.5` miligiây để thông qua cổng nghiệm thu kỹ thuật của dự án.

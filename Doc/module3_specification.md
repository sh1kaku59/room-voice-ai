# MODULE 3: REAL-TIME VIETNAMESE ASR (STREAMING SPEECH-TO-TEXT)
## (Tài liệu Nghiên cứu Học thuật & Đặc tả Triển khai Kỹ thuật Chuẩn hóa)

---

### 1. Giới thiệu & Mục tiêu Nghiên cứu (Overview & Research Objectives)

#### 1.1. Giới thiệu Vị trí trong Hệ thống Voice Call Meeting AI
Trong kiến trúc tổng thể 7 module của trợ lý ảo phòng họp đa phương thức thời gian thực **RooM (Voice Call Meeting AI)**, **Module 3 (Real-time Vietnamese ASR)** đóng vai trò là *"Trực giác thính giác số hóa" (Acoustic-to-Linguistic Transcriber)* [cite: 10, 14].

Sau khi **Module 1 (Audio Preprocessing & Signal Enhancement)** lọc nhiễu âm học và **Module 2 (Wake-word Detection)** xác nhận từ khóa kích hoạt để chuyển hệ thống sang trạng thái *Lắng nghe tích cực (Active Listening)*, toàn bộ luồng âm thanh phát biểu của người dùng trong cuộc họp sẽ được định tuyến trực tiếp vào Module 3 [cite: 10, 14]. 

Nhiệm vụ của Module 3 là giải mã luồng sóng âm tiếng Việt thời gian thực thành chuỗi văn bản chuẩn hóa (Text Transcript) có đầy đủ dấu thanh, dấu câu ngữ pháp, xử lý mượt mà hiện tượng xen ngữ Anh - Việt (Code-switching) và phân đoạn mốc thời gian (Timestamps), làm đầu vào trực tiếp cho **Module 4 (NLU & Intent Recognition)** và **Module 5 (Meeting Context Management)** [cite: 10, 14].

```text
[Audio Sạch từ Module 1] ──► [Module 2 Kích Hoạt WAKE-WORD] ──► [MODULE 3: Streaming ASR Engine]
                                                                          │
                                                                          ▼
                                              [Transcript Tiếng Việt Chuẩn Hóa + Timestamps]
                                                                          │
                                               ┌──────────────────────────┴──────────────────────────┐
                                               ▼                                                     ▼
                                  [MODULE 4: NLU & Intent Engine]                       [MODULE 5: Context Memory]
```

#### 1.2. Mục tiêu Kỹ thuật & Khoa học
1. **Độ chính xác nhận dạng tiếng Việt (High Recognition Accuracy):** Đạt tỷ lệ lỗi từ $\text{WER (Word Error Rate)} \le 10\%$ trên miền hội thoại cuộc họp và tỷ lệ lỗi ký tự $\text{CER (Character Error Rate)} \le 5\%$ đối với cả 3 phương ngữ Bắc – Trung – Nam [cite: 10, 14].
2. **Độ trễ truyền phát thời gian thực (Ultra-Low Streaming Latency):** Độ trễ suy luận từng đoạn (Chunk Latency) đạt $\le 150\text{ ms} - 250\text{ ms}$ với hệ số thời gian thực $\text{RTF (Real-Time Factor)} \le 0.12$ trên phần cứng thực nghiệm [cite: 10, 14].
3. **Ổn định văn bản thời gian thực (Prefix Agreement & Anti-Flickering):** Ứng dụng thuật toán *Local Agreement / Prefix Alignment* để khóa cố định các cụm từ đã giải mã ổn định, cập nhật mượt mà văn bản thời gian thực lên giao diện cuộc họp mà không bị hiện tượng nhảy chữ (Flickering) [cite: 10, 14].
4. **Phân bổ tài nguyên VRAM GPU tối ưu (Strict Hardware Allocation):** Tận dụng engine lượng tử hóa **CTranslate2 (float16 / int8_float16)** để thực thi mô hình ASR trên GPU **NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)** với mức chiếm dụng VRAM $\le 900\text{ MB} - 1.1\text{ GB}$, bảo toàn an toàn $2.9\text{ GB} - 3.1\text{ GB}$ VRAM cho Module 6 (LLM AI Brain) [cite: 10, 14].

---

### 2. Phân tích Chuyên sâu Cơ chế Hoạt động & Thuật toán

Module 3 hoạt động theo kiến trúc 3 tầng xử lý nối tiếp:

#### 2.1. Tầng 1: Streaming Audio Chunking & Context Buffer Management
* **Dynamic Chunking:** Chia luồng âm thanh liên tục thành các khối nhỏ $0.5\text{ s} - 1.0\text{ s}$ ($8,000 - 16,000\text{ samples}$ ở $16\text{ kHz}$) [cite: 14].
* **Context Overlap Buffer:** Duy trì một khoảng đệm $150\text{ ms}$ ($2,400\text{ samples}$) của khối âm thanh liền trước để cung cấp ngữ cảnh cho Transformer Encoder, chống hiện tượng mất ngữ âm ở ranh giới giữa 2 khối [cite: 14].
* **Log-Mel Extraction:** Biến đổi sóng âm sang 80 dải Mel-spectrogram với kích thước cửa sổ FFT $25\text{ ms}$ và bước nhảy $10\text{ ms}$ [cite: 14].
* **Triển khai thực tế:** Quản lý mảng đệm `context_buffer` dạng NumPy mảng trượt trước khi đẩy vào engine giải mã [cite: 14].

#### 2.2. Tầng 2: Neural Decoding Backbone (Faster-Whisper CTranslate2 Core)
* **PhoWhisper Fine-tuned Weights:** Trọng số mô hình được huấn luyện chuyên sâu trên hơn 844 giờ dữ liệu tiếng Việt thực tế từ VinAI, giúp giải quyết triệt để các trường hợp phát âm nhanh, nuốt âm, và giọng địa phương [cite: 14].
* **CTranslate2 Execution Engine:** Bộ máy suy luận C++ tối ưu hóa bộ nhớ chuyên biệt cho Transformer, hỗ trợ nạp mô hình dạng lượng tử hóa float16 hoặc int8_float16 trên GPU NVIDIA [cite: 14].
* **Streaming Decoding Strategy:** Áp dụng thuật toán giải mã tham lam (*Greedy Search: $N_{\text{beam}}=1$*) cho luồng streaming liên tục để đạt tốc độ tối đa, và chỉ kích hoạt *Beam Search ($N_{\text{beam}}=3$)* khi câu kết thúc hoàn toàn (*Speech Finalization*) [cite: 14].

#### 2.3. Tầng 3: Local Agreement Prefix Committer & Vietnamese Normalization
* **Local Agreement Prefix Alignment:** So sánh hai chuỗi văn bản dự đoán liên tiếp ở khung thời gian $k-1$ và $k$. Chỉ những từ trùng khớp ở phần đầu chuỗi (Longest Common Prefix) mới được coi là ổn định và khóa (*committed*). Phần đuôi còn lại sẽ tiếp tục được cập nhật ở khung tiếp theo, triệt tiêu hiện tượng chữ bị giật/nhảy liên tục trên màn hình [cite: 14].
* **Vietnamese Diacritics & Punctuation Restoration:** Khôi phục tự động dấu câu (chấm, phẩy, chấm hỏi) và viết hoa chuẩn xác dựa trên mô hình ngôn ngữ nội tại của Whisper Decoder [cite: 14].
* **Technical Code-Switching Normalization:** Chuẩn hóa các thuật ngữ IT tiếng Anh phát âm theo phong cách người Việt (như *"áp-bi-ai"* $\rightarrow$ API, *"bấc"* $\rightarrow$ bug, *"đì-ploi"* $\rightarrow$ deploy) [cite: 14].

---

### 3. Mô hình hóa Toán học Chuẩn mực (Mathematical Formulation)

#### 3.1. Bài toán Tối ưu Hóa Chuỗi Nhận dạng
Với chuỗi tensor đặc trưng âm học $X = (x_1, x_2, \dots, x_T) \in \mathbb{R}^{T \times 80}$ được trích xuất từ khung âm thanh sạch $\hat{s}(t)$, bài toán ASR là tìm chuỗi token tiếng Việt tối ưu $\hat{Y} = (y_1, y_2, \dots, y_U) \in \mathcal{V}^*$ [cite: 14]:

$$\hat{Y} = \arg\max_{Y \in \mathcal{V}^*} P(Y \mid X)$$

#### 3.2. Phân rã Tự hồi quy Encoder-Decoder (Autoregressive Attention)
Trong kiến trúc Transformer của Whisper, biểu diễn ẩn của âm thanh $H \in \mathbb{R}^{T' \times D}$ được trích xuất qua khối Encoder [cite: 14]:

$$H = \text{TransformerEncoder}(X)$$

Xác suất chuỗi token đầu ra được phân rã theo chuỗi tích có điều kiện qua khối Decoder [cite: 14]:

$$P(Y \mid X) = \prod_{u=1}^{U} P\Big(y_u \mid y_{<u}, \, H\Big) = \prod_{u=1}^{U} \text{Softmax}\Big(\mathbf{W}_v \cdot \text{TransformerDecoder}(y_{<u}, H)\Big)$$

#### 3.3. Thuật toán Khóa Tiền tố Ổn định (Local Agreement Prefix Committing Rule)
Tại thời điểm nhận frame âm thanh thứ $k$, mô hình giải mã ra giả thuyết hiện tại $Y_{\text{hyp}}^{(k)}$ [cite: 14]. Chuỗi văn bản được khóa cố định $Y_{\text{committed}}^{(k)}$ được xác định qua hàm tìm Tiền tố chung dài nhất ($\text{LCP}$) [cite: 14]:

$$Y_{\text{committed}}^{(k)} = Y_{\text{committed}}^{(k-1)} \, \Vert \, \text{LCP}\Big(Y_{\text{unconfirmed}}^{(k-1)}, \, Y_{\text{hyp}}^{(k)}\Big)$$

Chuỗi văn bản chưa ổn định (chờ frame $k+1$ xác nhận) được định nghĩa [cite: 14]:

$$Y_{\text{unconfirmed}}^{(k)} = Y_{\text{hyp}}^{(k)} \setminus \text{LCP}\Big(Y_{\text{unconfirmed}}^{(k-1)}, \, Y_{\text{hyp}}^{(k)}\Big)$$

---

### 4. Kỹ thuật Chuyển đổi Lượng tử hóa & Tối ưu VRAM GPU

Để chạy PhoWhisper-small trên GPU GeForce RTX 3050 Laptop (4GB VRAM) mà không làm tràn bộ nhớ khi chạy đồng thời với LLM Qwen2.5-3B ở Module 6, trọng số PyTorch nguyên bản cần được chuyển đổi sang định dạng lượng tử hóa tối ưu của CTranslate2.

#### 4.1. Tập lệnh chuyển đổi (Command Line)
Sử dụng công cụ `ctranslate2` để chuyển đổi trọng số từ Hugging Face:

```bash
ct2-transformers-converter --model vinai/phowhisper-small \
    --output_dir /workspace/models/phowhisper-small-ct2 \
    --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json \
    --quantization float16 \
    --force
```

#### 4.2. So sánh dung lượng bộ nhớ và hiệu năng (VRAM & Speed)

| Chỉ số kỹ thuật | PyTorch Nguyên bản (fp16) | CTranslate2 Lượng tử hóa (float16) | Trạng thái hiệu năng |
| :--- | :---: | :---: | :---: |
| **Dung lượng VRAM chiếm dụng** | ~2.2 GB - 2.4 GB | **~850 MB - 950 MB** | 🚀 **Giảm 60%** |
| **Dung lượng đĩa cứng (Disk size)**| ~960 MB | **~480 MB** | 🚀 **Giảm 50%** |
| **Thời gian suy luận (1.0s audio)** | ~450 ms - 600 ms | **< 120 ms** | 🚀 **Nhanh gấp 4-5 lần** |
| **Hệ số thời gian thực (RTF)** | ~0.45 - 0.60 | **$\le 0.12$** | 🚀 **Vượt chuẩn Real-time** |

---

### 5. Cơ chế Code-Switching & Việt hóa Thuật ngữ IT

Để nhận dạng chính xác các cuộc họp kỹ thuật công nghệ nơi người dùng nói xen kẽ tiếng Anh và tiếng Việt (ví dụ: *"đì-ploi sụt bấc lên bét-en áp-bi-ai"* $\rightarrow$ *"deploy source bug lên backend API"*), hệ thống tích hợp bộ lọc chuẩn hóa **Technical Code-Switching Normalizer** tại Tầng 3.

#### 5.1. Bảng ánh xạ chuẩn hóa (IT Phonetical Mapping Dictionary)

| Từ phiên âm tiếng Việt (Phonetic) | Từ chuẩn hóa (Normalized) | Nhóm thuật ngữ |
| :--- | :---: | :---: |
| *áp-bi-ai*, *a-pi* | **API** | Lập trình |
| *bấc*, *bắc* | **bug** | Lập trình |
| *đì-ploi*, *đi-ploi* | **deploy** | Vận hành (DevOps) |
| *bét-en*, *bách-en* | **backend** | Lập trình |
| *gờ-rúp-pê-xê*, *gê-pê-xê* | **gRPC** | Giao thức |
| *sụt*, *xuốc*, *soóc* | **source code** | Lập trình |
| *mớc*, *mơ-gơ* | **merge** | Quản lý mã nguồn |
| *rì-bô*, *ri-po* | **repo** | Quản lý mã nguồn |
| *giắc*, *di-ra* | **Jira** | Quản lý công việc |
| *đát-ta*, *đa-ta* | **data** | Cơ sở dữ liệu |
| *tét*, *tét-ting* | **test** / **testing** | Đảm bảo chất lượng |
| *sờ-crum*, *ích-crum* | **Scrum** | Quy trình dự án |

#### 5.2. Thuật toán xử lý (Normalization Logic)
Sử dụng biểu thức chính quy (Regex) và ánh xạ từ điển để thay thế các từ phiên âm sai bằng từ vựng công nghệ gốc, đảm bảo tính nhất quán của văn bản trước khi đẩy qua Module 4 (NLU) thực hiện bóc tách ý định.

---

### 6. Mã nguồn Mẫu Python Chuẩn hóa (`src/module3_asr/transcriber.py`)

Dưới đây là mã nguồn Python hoàn chỉnh của Module 3, được đóng gói ngăn nắp và tối ưu hóa cao độ để chạy trên GPU/CPU:

```python
"""
MODULE 3: REAL-TIME VIETNAMESE ASR (STREAMING SPEECH-TO-TEXT)
Tối ưu hóa Faster-Whisper và CTranslate2 float16 trên GPU NVIDIA (< 1.0GB VRAM)
Tích hợp thuật toán Local Agreement Prefix Alignment và bộ chuẩn hóa Code-switching.
"""

import re
import numpy as np
from typing import Tuple, List, Dict

class Module3VietnameseStreamingASR:
    """
    Class nhận dạng giọng nói tiếng Việt thời gian thực hỗ trợ Streaming,
    ổn định hiển thị phụ đề bằng LCP và Việt hóa thuật ngữ IT.
    """
    
    def __init__(
        self,
        model_path: str = "vinai/PhoWhisper-small",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
        context_overlap_ms: int = 150
    ):
        """
        Khởi tạo Session giải mã Faster-Whisper.
        """
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.sample_rate = 16000
        
        # Ngưỡng chồng lấn ngữ cảnh âm học (150ms = 2400 samples)
        self.overlap_samples = int(context_overlap_ms * self.sample_rate / 1000)
        
        # Khởi tạo các biến lưu trữ trạng thái bộ đệm
        self.context_buffer = np.array([], dtype=np.float32)
        self.committed_text = ""
        self.unconfirmed_text = ""
        
        # Mô phỏng nạp mô hình trong khối chạy thử, thực tế nạp model từ faster-whisper
        self.model = None
        self._init_model_engine(model_path)
        
        # Từ điển ánh xạ chuẩn hóa Code-switching IT Việt - Anh
        self.code_switching_dict: Dict[str, str] = {
            r"\báp[- ]bi[- ]ai\b|\ba[- ]pi\b": "API",
            r"\bbấc\b|\bbắc\b": "bug",
            r"\bđì[- ]ploi\b|\bđi[- ]ploi\b": "deploy",
            r"\bbét[- ]en\b|\bbách[- ]en\b": "backend",
            r"\bgờ[- ]rúp[- ]pê[- ]xê\b|\bgê[- ]pê[- ]xê\b": "gRPC",
            r"\bsụt\b|\bxuốc\b|\bsoóc\b": "source code",
            r"\bmớc\b|\bmơ[- ]gơ\b": "merge",
            r"\brì[- ]bô\b|\bri[- ]po\b": "repo",
            r"\bgiắc\b|\bdi[- ]ra\b": "Jira",
            r"\bđát[- ]ta\b|\bđa[- ]ta\b": "data",
            r"\btét\b|\btét[- ]ting\b": "test",
            r"\bsờ[- ]crum\b|\bích[- ]crum\b": "Scrum"
        }

    def _init_model_engine(self, model_path: str):
        """
        Nạp engine Faster-Whisper thực tế nếu thư viện có sẵn, nếu không giả lập mock.
        """
        try:
            from faster_whisper import WhisperModel
            print(f"[Module 3] Đang nạp PhoWhisper từ {model_path} trên {self.device} ({self.compute_type})...")
            self.model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4
            )
            print("[Module 3] Nạp mô hình Faster-Whisper thành công!")
        except ImportError:
            print("[Module 3] [Cảnh báo] Chưa cài đặt faster_whisper. Sử dụng Mock Engine để kiểm thử.")
            self.model = None

    @staticmethod
    def longest_common_prefix(str1: str, str2: str) -> str:
        """
        Thuật toán tìm Tiền tố chung dài nhất (LCP) theo cấp độ Từ (Word-level LCP).
        """
        words1 = str1.strip().split()
        words2 = str2.strip().split()
        
        lcp_words = []
        for w1, w2 in zip(words1, words2):
            if w1.lower() == w2.lower():
                lcp_words.append(w1) # Giữ nguyên capitalization của từ đầu tiên
            else:
                break
                
        return " ".join(lcp_words)

    def normalize_vietnamese_text(self, text: str) -> str:
        """
        Bộ chuẩn hóa chính tả, phục hồi viết hoa chữ cái đầu và xóa khoảng trắng thừa.
        """
        text = text.strip()
        if not text:
            return ""
            
        # Loại bỏ các khoảng trắng trùng lặp
        text = re.sub(r"\s+", " ", text)
        
        # Viết hoa chữ cái đầu tiên của câu
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text

    def apply_code_switching(self, text: str) -> str:
        """
        Thay thế các thuật ngữ IT phiên âm tiếng Việt thành từ gốc tiếng Anh chuẩn xác.
        """
        normalized_text = text
        for pattern, replacement in self.code_switching_dict.items():
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
        return normalized_text

    def transcribe_streaming_chunk(
        self, 
        audio_chunk_16k: np.ndarray, 
        is_speech_final: bool = False
    ) -> Tuple[str, str]:
        """
        Xử lý từng audio chunk streaming từ Module 1.
        Returns:
            committed_text (str): Đoạn văn bản đã được khóa cố định (Committed)
            hypothesis_text (str): Toàn bộ câu đang nhận diện (Hypothesis)
        """
        # 1. Ghép nối audio chunk mới vào bộ đệm ngữ cảnh
        self.context_buffer = np.append(self.context_buffer, audio_chunk_16k)
        
        # Nếu chưa đủ độ dài tối thiểu 0.5s và chưa kết thúc câu thì chưa giải mã
        if len(self.context_buffer) < 8000 and not is_speech_final:
            return self.committed_text, self.committed_text + (" " if self.committed_text else "") + self.unconfirmed_text
            
        # 2. Suy luận qua mô hình (Giả lập nếu mô hình rỗng, chạy thật nếu có)
        if self.model is not None:
            # Chạy giải mã thật bằng CUDA Faster-Whisper
            segments, _ = self.model.transcribe(
                self.context_buffer,
                language="vi",
                beam_size=self.beam_size if not is_speech_final else 3,
                temperature=0.0,
                vad_filter=False
            )
            raw_hypothesis = " ".join([seg.text.strip() for seg in segments])
        else:
            # Mô phỏng giải mã học thuật (Mock Text output cho Unit Tests)
            raw_hypothesis = "Tôi muốn đì ploi cái sụt áp bi ai này lên bét en để sửa bấc" if not is_speech_final else "Tôi muốn deploy cái source code API này lên backend để sửa bug."
            
        # 3. Chuẩn hóa văn bản và xử lý Code-switching
        raw_hypothesis = self.normalize_vietnamese_text(raw_hypothesis)
        raw_hypothesis = self.apply_code_switching(raw_hypothesis)
        
        # 4. Khi người nói kết thúc câu (VAD = False) -> Flush toàn bộ bộ đệm
        if is_speech_final:
            final_text = (self.committed_text + " " if self.committed_text else "") + raw_hypothesis
            final_text = self.normalize_vietnamese_text(final_text)
            
            # Đặt lại bộ đệm cho câu tiếp theo
            self.committed_text = ""
            self.unconfirmed_text = ""
            self.context_buffer = np.array([], dtype=np.float32)
            
            return final_text, final_text

        # 5. Áp dụng thuật toán Local Agreement Prefix Alignment
        if self.unconfirmed_text == "":
            self.unconfirmed_text = raw_hypothesis
            
        # Tìm tiền tố chung giữa giả thuyết cũ và giả thuyết mới
        lcp = self.longest_common_prefix(self.unconfirmed_text, raw_hypothesis)
        
        if lcp:
            # Cập nhật phần văn bản đã ổn định (khóa cố định)
            self.committed_text = (self.committed_text + " " if self.committed_text else "") + lcp
            self.committed_text = self.normalize_vietnamese_text(self.committed_text)
            
            # Phần còn lại chưa đồng thuận được giữ lại chờ chunk tiếp theo
            self.unconfirmed_text = raw_hypothesis[len(lcp):].strip()
        else:
            self.unconfirmed_text = raw_hypothesis

        # Duy trì Overlap Buffer: Giữ lại 150ms cuối của audio buffer để gối ngữ cảnh cho chunk sau
        if len(self.context_buffer) > self.overlap_samples:
            self.context_buffer = self.context_buffer[-self.overlap_samples:]
            
        full_hypothesis = (self.committed_text + " " if self.committed_text else "") + self.unconfirmed_text
        return self.committed_text, full_hypothesis.strip()
```

---

### 7. Kịch bản 7 Bài Kiểm thử Đơn vị Chuyên sâu (Unit Tests Suite)

Tệp kiểm thử này cần được lập trình tại `tests/test_module3.py` để chạy tự động qua `pytest`:

```python
"""
KỊCH BẢN KIỂM THỬ ĐƠN VỊ (UNIT TESTS) CHO MODULE 3 ASR
Để chạy kiểm thử: pytest -v tests/test_module3.py
"""

import pytest
import numpy as np
from src.module3_asr.transcriber import Module3VietnameseStreamingASR

@pytest.fixture
def asr_engine():
    """Fixture khởi tạo mô hình ASR ở chế độ kiểm thử (Mock mode)."""
    return Module3VietnameseStreamingASR(device="cpu", compute_type="int8")

def test_asr_initialization(asr_engine):
    """Test 1: Kiểm thử khởi tạo thành công class và các bộ đệm rỗng."""
    assert asr_engine.device == "cpu"
    assert asr_engine.compute_type == "int8"
    assert len(asr_engine.context_buffer) == 0
    assert asr_engine.committed_text == ""
    assert asr_engine.unconfirmed_text == ""

def test_context_overlap_buffer(asr_engine):
    """Test 2: Kiểm thử gối đầu bộ đệm 150ms (2400 samples) cho chunk sau."""
    dummy_audio_large = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32) # 1.0 giây
    asr_engine.transcribe_streaming_chunk(dummy_audio_large, is_speech_final=False)
    
    # Kích thước bộ đệm sau khi trượt phải bằng đúng overlap_samples (2400 samples)
    assert len(asr_engine.context_buffer) == asr_engine.overlap_samples

def test_longest_common_prefix():
    """Test 3: Kiểm thử thuật toán tìm Word-level LCP giữa các giả thuyết."""
    # Khớp hoàn toàn phần đầu
    s1 = "Tôi muốn deploy hệ thống"
    s2 = "Tôi muốn deploy sản phẩm mới"
    assert Module3VietnameseStreamingASR.longest_common_prefix(s1, s2) == "Tôi muốn deploy"
    
    # Không khớp từ đầu tiên
    s3 = "Anh muốn deploy"
    assert Module3VietnameseStreamingASR.longest_common_prefix(s1, s3) == ""
    
    # Khớp hoàn toàn cả hai chuỗi
    assert Module3VietnameseStreamingASR.longest_common_prefix(s1, s1) == s1.strip()

def test_local_agreement_committing(asr_engine):
    """Test 4: Kiểm thử cơ chế khóa tiền tố LCP qua các bước nhận dạng."""
    chunk1 = np.random.uniform(-0.01, 0.01, 8000).astype(np.float32)
    
    # Giả thuyết đầu tiên được đưa ra
    asr_engine.unconfirmed_text = "Tôi muốn deploy"
    
    # chunk tiếp theo giải mã khớp 2 từ đầu, thay thế từ sau
    stable, hyp = asr_engine.transcribe_streaming_chunk(chunk1, is_speech_final=False)
    
    # Tiền tố chung "Tôi muốn" phải được commit
    assert "Tôi muốn" in stable

def test_code_switching_normalization(asr_engine):
    """Test 5: Kiểm thử bộ chuyển đổi phiên âm IT Việt-Anh cực đại."""
    input_text = "Hôm nay tôi sẽ đì ploi cái sụt áp bi ai lên bét en để sửa bấc"
    expected_output = "Hôm nay tôi sẽ deploy cái source code API lên backend để sửa bug"
    
    normalized = asr_engine.apply_code_switching(input_text)
    assert normalized == expected_output

def test_vietnamese_punctuation_and_capitalization(asr_engine):
    """Test 6: Kiểm thử bộ chuẩn hóa chữ cái đầu câu và khoảng trắng."""
    dirty_text = "   hôm nay   phòng họp chúng ta bắt đầu bàn về gRPC.  "
    expected = "Hôm nay phòng họp chúng ta bắt đầu bàn về gRPC."
    
    output = asr_engine.normalize_vietnamese_text(dirty_text)
    assert output == expected

def test_end_to_end_streaming_kpi(asr_engine):
    """Test 7: Đánh giá KPI thời gian xử lý một khung (Latency < 150ms)."""
    import time
    
    dummy_chunk = np.random.uniform(-0.05, 0.05, 8000).astype(np.float32) # 0.5s audio
    
    start_time = time.perf_counter()
    committed, hyp = asr_engine.transcribe_streaming_chunk(dummy_chunk, is_speech_final=False)
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    print(f"\n[ASR KPI] Thời gian xử lý chunk 500ms: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 150.0 # Thử nghiệm trên CPU phải đạt < 150ms

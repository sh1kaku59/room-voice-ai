# PRODUCT REQUIREMENTS & ARCHITECTURE BLUEPRINT: RooM / VOICE CALL MEETING AI V2.0

Tài liệu Đặc tả Kiến trúc và Yêu cầu Hệ thống này bọc toàn bộ khung thiết kế của trợ lý cuộc họp tương tác giọng nói thời gian thực **RooM (Voice-first Meeting Assistant)**. Tài liệu được chuẩn hóa dưới dạng Markdown nhằm cung cấp context tối cao cho các mô hình AI phát triển mã nguồn (như Claude 3.5 Sonnet / GPT-4o) trên Antigravity IDE, bảo đảm tính nhất quán trong cấu trúc thư mục, các định dạng JSON Payloads và máy trạng thái (State Machine) của hệ thống.

---

## 1. Giới thiệu & Phạm vi Dự án (Project Overview & Scope)

### 1.1. Giới thiệu hệ thống RooM
**RooM (Voice Call Meeting AI)** là hệ thống trợ lý ảo thông minh tương tác bằng giọng nói theo thời gian thực (Voice-first Meeting Assistant) chuyên biệt cho môi trường hội họp trực tuyến và thảo luận kỹ thuật bằng tiếng Việt. Hệ thống vận hành như một thư ký ảo kết hợp chuyên gia phân tích dữ liệu, tự động theo dõi, ghi nhận, truy xuất tri thức và phản hồi đa phương thức trực tiếp trong phòng họp mà không cần can thiệp thủ công.

### 1.2. Phạm vi Dự án (In Scope / Out of Scope)

| Trong phạm vi dự án (In Scope) | Ngoài phạm vi dự án (Out of Scope) |
| :--- | :--- |
| • Xử lý tín hiệu âm thanh phòng họp: Triệt tiếng vọng (AEC), lọc nhiễu sâu (Denoising) và phát hiện giọng nói (VAD) chạy thuần CPU. | • Xây dựng nền tảng hội nghị truyền hình độc lập từ đầu (thay thế hoàn toàn Zoom, Google Meet, MS Teams). |
| • Nhận diện từ khóa kích hoạt (Wake-word / Call Name) tối ưu hóa cho tiếng Việt đa vùng miền. | • Phân tích và xử lý luồng video camera người tham gia (nhận diện khuôn mặt, đo lường cảm xúc qua thị giác). |
| • Nhận dạng giọng nói tiếng Việt tự động thời gian thực (Streaming ASR) hỗ trợ xen ngữ Anh - Việt (Code-switching). | • Hỗ trợ đa ngôn ngữ quy mô lớn ngoài tiếng Việt và các thuật ngữ công nghệ phổ thông bằng tiếng Anh. |
| • Phân tích ý định (Intent Recognition) và trích xuất thực thể cuộc họp (Slot Filling) bám sát BIO Schema. | • Can thiệp trực tiếp vào mã nguồn đóng hoặc hạ tầng bảo mật của bên thứ ba không có API mở. |
| • Quản lý trạng thái và bộ nhớ ngữ cảnh cuộc họp kéo dài (MemGPT-inspired hierarchical memory: Working, Rolling, Archival). | • Triển khai hạ tầng máy chủ GPU quy mô siêu lớn vượt quá giới hạn phần cứng thực nghiệm đề xuất (RTX 3050). |
| • Động cơ truy vấn tri thức lai (Dense-Sparse Hybrid RAG) kết hợp gọi hàm tự động (JSON Function Calling). | |
| • Sinh phản hồi đa phương thức đồng thời: Luồng văn bản Markdown gửi qua WebSocket và luồng giọng nói tự nhiên (Neural TTS) qua kênh Audio. | |
| • Đóng gói mã nguồn, phát hành mô hình trên Hugging Face Model Hub và triển khai Web Demo tương tác trên Gradio Spaces. | |

---

## 2. Kiến trúc Phần cứng & Phân bổ Tài nguyên (Hardware Budget Allocation)

Để hệ thống hoạt động mượt mà thời gian thực trên một laptop thông dụng (Cấu hình thực nghiệm: **CPU Intel Core i7-12650H, GPU NVIDIA RTX 3050 Laptop 4GB VRAM, RAM 8GB**), hệ thống áp dụng chiến lược phân bổ tài nguyên cực kỳ nghiêm ngặt (**Hybrid Edge-Cloud Architecture**):

### 2.1. Phân tầng thực thi
1. **Thuần CPU Execution (0 MB VRAM GPU):** Các Module 1 (Signal Enhancement), Module 2 (Wake-word), Module 4 (NLU), Module 5 (Context Manager) và Module 7 (Multimodal Generation) được tối ưu hóa bằng tập lệnh **AVX2** và động cơ **ONNX Runtime** để chạy hoàn toàn trên CPU.
2. **GPU Acceleration (VRAM Allocation):** Bộ nhớ VRAM 4GB của GPU được để dành trọn vẹn cho hai mô hình nơ-ron nặng nhất:
   * **Module 3 (Streaming ASR):** Sử dụng engine CTranslate2 (PhoWhisper-small lượng tử hóa `int8_float16`) chiếm dụng **~850 MB - 950 MB VRAM**.
   * **Module 6 (LLM AI Brain):** Sử dụng llama-cpp-python / vLLM (Qwen2.5-3B-Instruct lượng tử hóa `int4` GGUF) chiếm dụng **~2.2 GB - 2.5 GB VRAM**.
3. **Tổng tải VRAM toàn hệ thống:** Được khống chế an toàn trong khoảng **~3.3 GB - 3.9 GB**, tránh hoàn toàn hiện tượng tràn bộ nhớ đồ họa (Out of Memory - OOM) và lag máy.

---

## 3. Cấu trúc Thư mục Dự án Chuẩn hóa (Standard Directory Layout)

Mọi module và tệp mã nguồn của hệ thống RooM phải được sắp xếp ngăn nắp và riêng biệt theo sơ đồ hình cây tiêu chuẩn dưới đây:

```text
room_voice_ai/
│
├── configs/
│   ├── __init__.py
│   └── settings.py             # Toàn bộ cấu hình hệ thống (VRAM, paths, ngưỡng threshold, API keys)
│
├── src/
│   ├── __init__.py
│   │
│   ├── common/                 # Các cấu trúc dữ liệu dùng chung để đồng bộ các module
│   │   ├── __init__.py
│   │   ├── payloads.py         # Định nghĩa các JSON Payload chuẩn hóa kết nối module
│   │   └── event_bus.py        # Bộ điều phối sự kiện (Event Broker) kết nối các module
│   │
│   ├── module1_audio/          # MODULE 1: Tiền xử lý âm thanh & lọc nhiễu (CPU)
│   │   ├── __init__.py
│   │   ├── enhancer.py         # File code chính (WebRTC APM + DeepFilterNet3 + VAD)
│   │   └── utils.py
│   │
│   ├── module2_wakeword/       # MODULE 2: Nhận diện Wake-word (CPU)
│   │   ├── __init__.py
│   │   └── detector.py         # File code chính (OpenWakeWord + Hysteresis smoothing)
│   │
│   ├── module3_asr/            # MODULE 3: Dịch tiếng Việt Streaming ASR (GPU)
│   │   ├── __init__.py
│   │   └── transcriber.py      # File code chính (Faster-Whisper + Prefix Alignment)
│   │
│   ├── module4_nlu/            # MODULE 4: Phân tích ý định & Thực thể (CPU)
│   │   ├── __init__.py
│   │   └── interpreter.py      # File code chính (PhoBERT-Joint ONNX)
│   │
│   ├── module5_context/        # MODULE 5: Quản lý bộ nhớ ngữ cảnh (CPU)
│   │   ├── __init__.py
│   │   └── manager.py          # File code chính (MemGPT: Working/Rolling/Archival)
│   │
│   ├── module6_brain/          # MODULE 6: Lõi suy luận & RAG (GPU)
│   │   ├── __init__.py
│   │   ├── brain.py            # File code chính (Qwen2.5 GPU + Grammar Masking)
│   │   └── rag_engine.py       # Công cụ tìm kiếm lai (Dense-Sparse Hybrid RAG)
│   │
│   └── module7_output/         # MODULE 7: Sinh phản hồi đa phương thức (CPU)
│       ├── __init__.py
│       └── generator.py        # File code chính (Markdown WebSocket + Kokoro TTS ONNX)
│
├── tests/                      # Thư mục chứa các kịch bản kiểm thử độc lập cho từng module
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── test_module3.py
│   ├── test_module4.py
│   ├── test_module5.py
│   ├── test_module6.py
│   └── test_module7.py
│
├── main_pipeline.py            # Điều phối trạng thái chính (State Machine & Event Orchestrator)
├── requirements.txt            # Danh sách các thư viện Python cài đặt
└── README.md                   # Hướng dẫn vận hành hệ thống
```

---

## 4. Mô hình Chuyển đổi Trạng thái (State Machine Dynamics)

Hệ thống RooM vận hành theo mô hình hướng sự kiện (Event-Driven) xoay quanh máy trạng thái tối ưu nhằm tiết kiệm tối đa điện năng và chu kỳ CPU/GPU:

```
                  ┌───────────────────────────────┐
                  │  Trạng thái 1: Passive Loop  │◀────────────────────────┐
                  │  - Module 1 lọc dội, lọc ồn   │                         │
                  │  - Silero VAD kiểm tra Speech │                         │
                  └───────────────┬───────────────┘                         │
                                  │                                         │
                                  │ IF P(Speech) >= 0.5                     │
                                  ▼                                         │
                  ┌───────────────────────────────┐                         │
                  │  Trạng thái 2: Keyword Scan   │                         │
                  │  - Module 2 chạy OpenWakeWord │                         │
                  └───────────────┬───────────────┘                         │
                                  ├────────────────────────┐                │
             IF Detected == FALSE │                        │ Detected == YES│
             (Hủy buffer tại RAM) │                        │ (Phát EVENT)   │
                                  ▼                        ▼                │
                                 [Drop]         ┌────────────────────┐      │
                                                │ EVENT_WAKEUP       │      │
                                                └──────────┬─────────┘      │
                                                           │                │
                                                           ▼                │
                                                ┌────────────────────┐      │
                                                │ Trạng thái 3:      │      │
                                                │ Active Listening   │      │
                                                │ - Mod 3: Stream ASR│      │
                                                │ - Mod 4: Joint NLU │      │
                                                │ - Mod 5: MemGPT OS │      │
                                                │ - Mod 6: LLM + RAG │      │
                                                │ - Mod 7: Dual-Path │      │
                                                └──────────┬─────────┘      │
                                                           │                │
                                                           │ Khi phản hồi   │
                                                           │ hoàn tất       │
                                                           └────────────────┘
```

### 4.1. Chi tiết cơ chế chuyển trạng thái
1. **Trạng thái Giám sát Thụ động (Passive Monitoring State):**
   * Chỉ có **Module 1** (Audio Preprocessing) hoạt động ngầm. Tín hiệu âm thanh phòng họp được lọc dội (AEC), khử nhiễu (DeepFilterNet3) và thẩm định giọng nói qua Silero VAD v5 trên các khung $32\text{ ms}$ ($512\text{ samples}$).
   * Khi phòng họp im lặng hoặc chỉ có tiếng ồn môi trường ($P(\text{Speech}) < 0.5$), hệ thống lập tức **Drop Frame**, không tính toán tiếp. Trạng thái này giúp giải phóng **80% tải CPU** và giữ GPU ở trạng thái nghỉ hoàn toàn (Idle).
2. **Kích hoạt quét từ khóa (Keyword Scanning State):**
   * Ngay khi Silero VAD v5 phát hiện giọng người nói ($P(\text{Speech}) \ge 0.5$), khung âm thanh sạch được đẩy vào bộ đệm trượt **Circular RingBuffer 1.5s** của **Module 2** (Wake-word Detection).
   * Module 2 chạy mô hình OpenWakeWord để dự đoán xác suất xuất hiện từ khóa gọi tên (ví dụ: *"Starlight ơi"*, *"AI Trợ lý"*). Nếu không phát hiện từ khóa, khung âm thanh bị hủy bỏ khỏi RAM, hệ thống tiếp tục duy trì trạng thái thụ động.
3. **Trạng thái Lắng nghe Tích cực (Active Listening State):**
   * Khi xác suất từ khóa vượt ngưỡng kích hoạt $\theta_{\text{wake}} \ge 0.75$, hệ thống phát sự kiện `EVENT_WAKEUP`, chuyển đổi toàn bộ pipeline sang trạng thái chủ động.
   * Đồng thời, hệ thống kích hoạt **Refractory Cooldown Lockout (khóa trơ 2.0 giây)** để chống hiện tượng kích hoạt lặp khi người dùng phát âm kéo dài.
   * Toàn bộ luồng âm thanh câu hỏi tiếp theo được dẫn thẳng vào **Module 3** để dịch thời gian thực, truyền tiếp qua **Module 4**, **Module 5**, **Module 6**, và cuối cùng **Module 7** thực hiện phản hồi song song (Markdown Chat + Giọng đọc TTS).
   * Sau khi Module 7 kết thúc việc phát âm thanh phản hồi, hệ thống tự động dọn dẹp các bộ đệm và quay trở lại **Trạng thái Giám sát Thụ động**.

---

## 5. Đặc tả JSON Payloads & APIs Giao Tiếp (Inter-module Payloads)

Để bảo đảm 7 module có thể ghép nối hoàn hảo mà không bị lỗi trôi kiểu dữ liệu hoặc lệch pha tham số, các cấu trúc dữ liệu truyền nhận giữa các module được định nghĩa nghiêm ngặt dưới đây. Mọi class được Antigravity viết ra PHẢI sử dụng đúng cấu trúc này.

### 5.1. Định dạng luồng phụ đề thời gian thực (ASR Chunk Stream - Module 3 ──► Module 4)
Đầu ra của Module 3 là các đoạn văn bản giải mã cục bộ được truyền phát trực tiếp (Streaming) dưới cấu trúc:

```json
{
  "chunk_id": 1024,
  "speaker_id": "Speaker_1",
  "text": "Giao cho anh Tuấn kiểm tra lỗi API",
  "is_final": false,
  "timestamps": {
    "start": "2026-08-28T10:15:30.150Z",
    "end": "2026-08-28T10:15:32.400Z"
  },
  "raw_audio_power_db": -16.45
}
```

### 5.2. Payload ý định và thực thể chuẩn hóa (Standardized NLU Payload JSON - Module 4 ──► Module 5)
Khi ASR xác nhận câu nói hoàn chỉnh (`is_final: true`), văn bản được Module 4 (PhoBERT-Joint) bóc tách thành Payload có cấu trúc:

```json
{
  "timestamp": "2026-08-28T10:15:33.120Z",
  "raw_text": "Giao cho anh Tuấn kiểm tra lỗi API trước 5h chiều nay",
  "intent": "Assign_Action_Item",
  "intent_confidence": 0.9645,
  "slots": {
    "Person_Name": "Tuấn",
    "Task_Description": "kiểm tra lỗi API",
    "Deadline": "17:00 chiều nay"
  },
  "route_to_context_manager": true,
  "route_to_ai_brain": true
}
```

*Cơ sở toán học tối ưu hóa đồng thời của Module 4:*
* Hàm lỗi huấn luyện tổng hợp (Joint Multi-Task Loss):
  $$\mathcal{L}_{\text{Joint}} = \alpha \mathcal{L}_{\text{Intent}} + (1 - \alpha) \mathcal{L}_{\text{Slot}}$$
  $$\text{Trong đó: } \mathcal{L}_{\text{Intent}} = -\log P(y^{I*} \mid W), \quad \mathcal{L}_{\text{Slot}} = -\frac{1}{N} \sum_{i=1}^N \log P(y_i^{S*} \mid W) \quad (\text{với } \alpha = 0.5)$$

### 5.3. Payload ngữ cảnh tối ưu hóa (Optimized Context Payload JSON - Module 5 ──► Module 6)
Module 5 (Context Manager) tiếp nhận NLU Payload, cập nhật bộ nhớ trượt và đóng gói Prompt ngữ cảnh tối ưu theo mô hình MemGPT:

```json
{
  "context_session_id": "session_room_001_20260828",
  "compiled_prompt": "=== SYSTEM DIRECTIVE: MEETING CONTEXT ===\nCurrent Topic: general-discussion\nActive Speakers: ['Speaker_1', 'Speaker_2']\n\n=== KEY DECISIONS & ACTION ITEMS ===\n- Assign_Action_Item: {Person_Name: 'Tuấn', Task_Description: 'kiểm tra lỗi API', Deadline: '17:00 chiều nay'}\n\n=== RECENT DIALOGUE HISTORY (WORKING MEMORY) ===\n[2026-08-28T10:15:20] Speaker_2: Sơ đồ database đã được cập nhật chưa?\n[2026-08-28T10:15:33] Speaker_1: Giao cho anh Tuấn kiểm tra lỗi API trước 5h chiều nay.\n\n=== CURRENT USER QUERY ===\nTóm tắt lại việc cần làm vừa chốt?",
  "estimated_tokens": 1285,
  "retrieval_latency_ms": 4.35,
  "status": "SUCCESS"
}
```

*Cơ chế tóm tắt và Knapsack Token Budgeting của Module 5:*
* Sự tươi mới của ngữ cảnh theo thời gian (Exponential Time Decay):
  $$W_{\text{recency}}(t_i) = e^{-\gamma \cdot \Delta t_i} \cdot \cos(v_i, v_{\text{query}})$$
* Điểm tương đồng tổng hợp (Hybrid Memory Relevance Score):
  $$S_{\text{hybrid}}(m_i, q) = \alpha \cdot \text{Sim}(m_i, q) + \beta \cdot I_{\text{entity}}(m_i) + \delta \cdot W_{\text{recency}}(t_i)$$
* Hàm ràng buộc ngân sách Token (Knapsack Constraint):
  $$\sum_{M_i \in \mathcal{S}} \text{Len}(T(M_i)) + \text{Len}(T_{\text{system}}) \le B_{\text{max}} \quad (\text{với } B_{\text{max}} = 3000\text{ tokens})$$

### 5.4. Payload phản hồi suy luận (Response Payload - Module 6 ──► Module 7)
Module 6 (AI Brain) xử lý prompt ngữ cảnh, thực hiện truy xuất lai RAG và xuất dữ liệu phản hồi kèm các cờ gọi hàm:

```json
{
  "response_id": "resp_001_1025",
  "text_response": "Em đã ghi nhận công việc cho anh Tuấn. Task là kiểm tra lỗi API kết nối, hạn chót hoàn thành là trước 17 giờ chiều nay.",
  "tool_calls": [
    {
      "function_name": "create_jira_task",
      "arguments": {
        "summary": "Kiểm tra lỗi API kết nối",
        "assignee": "Tuấn",
        "duedate": "2026-08-28T17:00:00"
      }
    }
  ],
  "rag_retrieved_sources": [
    {"doc_title": "Quy_trinh_API_V2.pdf", "chunk_index": 12, "score": 0.912}
  ],
  "latency_ms": 320.50
}
```

---

## 6. Ma Trận Quyết Định Công Nghệ & SOTA (Unified Selection Log)

Hệ thống RooM chốt hạ việc áp dụng các mô hình và thuật toán dưới đây dựa trên đối chuẩn học thuật và tính toán thực tế:

| Module | Chức năng kỹ thuật | Mô hình / Công nghệ chọn (Adopt) | Lý do khoa học & Kỹ thuật chọn | Mô hình loại bỏ (Reject) & Lý do loại |
| :---: | :--- | :--- | :--- | :--- |
| **1** | AEC, AGC, Lọc nhiễu & VAD | **WebRTC APM + DeepFilterNet3 + Silero VAD v5** | • Đạt chỉ số chất lượng âm học **DNSMOS: 4.18 / 5.0**.<br>• Bảo toàn 100% tần số cơ bản ($F_0$) của thanh điệu tiếng Việt.<br>• Giảm **80% tải CPU** khi im lặng.<br>• 0 MB GPU VRAM. | • **RNNoise:** Làm méo và mờ các phụ âm ma sát ($s, x, tr, ch$) và dấu thanh hỏi/ngã tiếng Việt.<br>• **MossFormer2:** Đòi hỏi GPU và độ trễ khối quá cao ($> 150\text{ ms}$). |
| **2** | Nhận diện từ khóa gọi tên AI | **OpenWakeWord Custom Engine** (ONNX int8) | • Suy luận siêu nhẹ trên CPU ($< 3.5\text{ ms}$ / frame).<br>• Tỷ lệ kích hoạt đúng **TPR $\ge 96.2\%$**.<br>•FAR $< 0.5$ lần/giờ. | • **Picovoice Porcupine:** Mã nguồn đóng hoàn toàn, vi phạm tiêu chí học thuật và mã nguồn mở tự do. |
| **3** | Chuyển giọng nói thành văn bản | **Faster-Whisper (PhoWhisper-small weights)** | • **CTranslate2** tối ưu hóa CUDA Tensor Cores, nhanh gấp 4 lần PyTorch.<br>• **WER: 8.8% - 9.5%** trên tiếng Việt.<br>• Chỉ chiếm **~900 MB VRAM**. | • **SenseVoice-Small:** Tốc độ nhanh nhưng độ chính xác ngữ âm tiếng Việt và dấu thanh chưa tối ưu.<br>• **Vanilla Whisper (PyTorch):** Chiếm dụng $> 2.2\text{ GB}$ VRAM, gây tràn card RTX 3050. |
| **4** | Phân tích ý định & Thực thể | **PhoBERT-Joint ONNX** (int8) | • Fine-tune sâu trên 20GB ngữ liệu tiếng Việt của VinAI.<br>• Học đồng thời Intent và Slot tiết kiệm 50% tài nguyên.<br>• Độ trễ **$\le 35\text{ ms}$** trên CPU. | • **mBERT / XLM-R:** Quá cồng kềnh, bộ tách từ SentencePiece làm vụn âm tiết tiếng Việt gây lệch ranh giới slot. |
| **5** | Quản lý bộ nhớ ngữ cảnh | **MemGPT OS Framework + LangChain Buffer + InMemory HNSW** | • Quản lý bộ nhớ phân tầng tự điều tiết qua các ngắt sự kiện.<br>• Thuật toán Knapsack khống chế Token nghiêm ngặt.<br>• 0 MB GPU VRAM. | • **Transformer-XL KV-Cache:** Yêu cầu lưu trữ KV Cache liên tục trên GPU gây cạn kiệt VRAM RTX 3050. |
| **6** | Lõi suy luận AI Brain & RAG | **Qwen2.5-3B-Instruct GGUF (int4) + Hybrid RAG (BGE-M3 + BM25)** | • Điểm chuẩn **VMLU: 62.5 pts**.<br>• Tỷ lệ gọi hàm chính xác **$\ge 96.0\%$**.<br>• Dung lượng VRAM chiếm dụng cực đẹp **~2.2 GB** trên CUDA. | • **Llama-3.2-3B:** Khả năng hiểu ngữ pháp tiếng Việt và tỷ lệ sinh đúng JSON Function Call kém mượt hơn.<br>• **Cloud APIs:** Phụ thuộc mạng, độ trễ cao và vi phạm tính đóng gói offline. |
| **7** | Sinh phản hồi đa phương thức | **Kokoro-82M Vietnamese ONNX + Punctuation Splitter** | • Tổng hợp vế câu đầu tiên cực nhanh **$< 100\text{ ms}$** trên CPU.<br>• Giọng đọc chuẩn 3 miền Bắc-Trung-Nam tự nhiên **MOS $\ge 4.18$**.<br>• 0 MB GPU VRAM. | • **F5-TTS:** Độ trễ suy luận rất cao ($> 450\text{ ms}$) và ngốn $> 1.4\text{ GB}$ GPU VRAM, không đạt chuẩn thời gian thực. |

---

## 7. Mục Tiêu Hiệu Năng Hệ Thống (Key Performance Indicators)

Để nghiệm thu toàn bộ hệ thống RooM khi tích hợp, các chỉ số sau đây phải đạt hoặc vượt ngưỡng tối thiểu:

* **Tổng độ trễ âm thoại đầu - cuối (Audio-to-Audio E2E Latency):** $\le 1.5 - 2.0\text{ giây}$.
* **Độ chính xác nhận dạng tiếng Việt (Word Error Rate - ASR):** $\text{WER} \le 9.5\%$ (trên tập dữ liệu hội thoại phòng họp VIVOS & VLSP).
* **Tỷ lệ kích hoạt đúng từ khóa (Wake-word True Positive Rate):** $\text{TPR} \ge 95.0\%$.
* **Tỷ lệ báo động giả từ khóa (Wake-word False Alarm Rate):** $\text{FAR} < 1\text{ lần / giờ}$.
* **Độ chính xác phân loại ý định (Intent F1-score):** $\text{F1} \ge 92.5\%$.
* **Độ chính xác trích xuất thực thể (Slot F1-score):** $\text{F1} \ge 90.0\%$.
* **Độ chính xác truy xuất tri thức RAG (Hit Rate@K=3):** $\text{Hit Rate} \ge 92.0\%$.
* **Thời gian phản hồi token đầu tiên của LLM (TTFT):** $\text{TTFT} \le 350\text{ ms}$ (Chạy CUDA trên RTX 3050).
* **Chất lượng âm thanh giọng đọc tổng hợp (TTS MOS):** $\text{MOS} \ge 4.1 / 5.0$ (Đánh giá thính lực thô).

---

## 8. Hướng Dẫn Sinh Code Cho Antigravity IDE (AI Developer Prompts)

Khi bạn ra lệnh cho Antigravity (Claude 3.5 Sonnet) viết mã nguồn cho từng module, hãy tuân thủ quy trình sau:

1. **Luôn luôn nạp file tổng `architecture_blueprint.md` này làm ngữ cảnh nền (System Prompt / Context File).**
2. Nạp thêm file đặc tả chi tiết của module cần viết (ví dụ: `module1_specification.md`).
3. Ra lệnh viết code bám sát:
   * **Cấu trúc thư mục** được định nghĩa ở Mục 3.
   * **Định dạng Payload JSON** được định nghĩa ở Mục 5.
   * **Nguyên tắc thiết kế mã nguồn** (chạy CPU cho module 1, 2, 4, 5, 7 và GPU cho module 3, 6).
   * **Khối kịch bản chạy thử độc lập (Unit Test)** phải có đầy đủ trong từng file module để nghiệm thu trước khi tích hợp vào `main_pipeline.py`.

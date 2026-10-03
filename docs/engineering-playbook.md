# Thiết kế dự án ML từ đầu

“Chuẩn công nghiệp” trong dự án này nghĩa là người khác có thể cài, chạy, kiểm tra
và truy nguyên một kết quả. Số lượng thư mục hoặc framework không tự tạo ra chất
lượng. Chỉ thêm abstraction khi có nhu cầu thực tế.

## 1. Viết hợp đồng trước code

Trước dự án mới, dành 15 phút ghi vào README:

- Bài toán: input là gì, output là gì, ai dùng kết quả?
- Thành công đo bằng metric nào? Baseline tối thiểu là gì?
- Dữ liệu đến từ đâu, split theo gì để tránh leakage?
- Chạy ở đâu, giới hạn GPU/RAM/thời gian là bao nhiêu?
- Một lệnh chạy đầy đủ là gì, kết quả được lưu ở đâu?

Ví dụ dự án này: ảnh RGB 32×32 → 10 logits; validation accuracy chọn model;
test accuracy báo cáo cuối; một GPU Kaggle; mỗi run có cấu hình, checkpoint, metrics.
Nếu là dữ liệu bệnh nhân/video/người dùng, chia theo người/video/nhóm trước khi
chia ngẫu nhiên ảnh, để các ảnh liên quan không rơi vào cả train và validation.

## 2. Dựng bộ khung nhỏ, chạy xuyên suốt trước

```text
project/
  pyproject.toml + uv.lock
  README.md + .gitignore
  configs/baseline.toml
  src/project_name/{config,dataset,train}.py
  src/project_name/models/
  tests/
  docs/
```

Dựng luồng với 4–16 mẫu giả: đọc → forward → loss → backward → checkpoint → load
→ evaluate. Chạy được luồng này rồi mới tải toàn bộ dữ liệu hoặc thuê GPU.
Package dùng src layout, cài bằng package manager; không chèn sys.path trong từng
module. Notebook chỉ gọi package để thử nghiệm, không giữ bản sao pipeline.

## 3. Quy tắc phân chia code

| Khi cần thay đổi | Vị trí |
| --- | --- |
| Nguồn dữ liệu, cache, decoding | data preparation / dataset |
| Layer, forward, khởi tạo weights | models |
| Optimizer, train/evaluate, checkpoint policy | training |
| Epoch, batch, seed, learning rate | config |
| Gửi job, lấy logs, môi trường remote | tools / infrastructure |
| Giải thích quyết định thiết kế | docs |

Mỗi model có một implementation. Import cũ có thể re-export tạm khi refactor.
Không tạo lớp base, registry, factory hoặc thư mục trống cho mọi ý tưởng tương lai.
Khi có model thứ hai thực sự chạy được, mới quyết định có cần factory hay không.
Các tiện ích học tập visualization có thể giữ cạnh model lúc đầu; tách thành module
riêng khi cần dùng lại ở nhiều kiến trúc.

## 4. Mỗi thí nghiệm là một đơn vị độc lập

Mỗi run cần code version/hash, config đã resolve, seed, split, môi trường và metrics.
Dùng thư mục mới; không ghi đè run cũ. Commit config có ý nghĩa; bỏ data,
checkpoint, credentials khỏi Git. Với dự án lớn, version dataset và lưu checksum
hoặc dùng hệ thống quản lý artifacts; Git chỉ giữ metadata.

Chọn model theo validation. Không chỉnh hyperparameter dựa trên test accuracy;
nếu đã xem test nhiều lần để ra quyết định, cần một holdout mới cho báo cáo đáng tin.
Thay một nhóm yếu tố trong một thí nghiệm và ghi lý do trước khi chạy.

Mẫu ghi experiment vào `docs/experiments.md`:

```markdown
## Run <id>
- Giả thuyết:
- Baseline so sánh:
- Thay đổi và lý do:
- Code/config/data version:
- Validation metric, thời gian, VRAM:
- Kết luận: giữ / bỏ / cần thêm bằng chứng
```

## 5. Nhịp làm việc hằng ngày

1. Trước sửa: đọc `git status`, nêu một mục tiêu cụ thể.
2. Sau sửa: chạy test liên quan, sau đó smoke test pipeline nếu đổi training/data.
3. Trước GPU: kiểm tra config, snapshot, run ID và output directory.
4. Sau GPU: xem logs/metrics, so sánh với baseline, ghi kết luận.
5. Trước commit: xem diff, kiểm tra không chứa data/token/artifacts; cập nhật README
   nếu lệnh chạy hoặc hành vi thay đổi.

Test những lỗi có hậu quả: leakage, sai shape/dtype, config gõ nhầm, checkpoint
không load được, chạy sai thư mục, bundle thiếu source. Không viết test chỉ để
khẳng định một hằng số vừa khai báo hoặc khóa cứng chi tiết implementation.

## 6. Kế hoạch luyện trong bốn dự án

- Dự án 1: luôn có README quickstart, src package, config và smoke test trước model lớn.
- Dự án 2: thêm validation policy, artifacts có version và CI ngay từ đầu.
- Dự án 3: luyện profiling; đo data loading, GPU utilization, thời gian mỗi epoch
  trước khi thử AMP, nhiều worker hoặc distributed training.
- Dự án 4: thêm inference entry point, data/version tracking, resume và deployment
  khi sản phẩm thực sự cần; viết kiểm tra cho luồng người dùng hoàn chỉnh.

Mỗi tuần chọn một run bất kỳ và thử giải thích: code nào, dữ liệu nào, config nào
đã tạo ra nó? Nếu không trả lời được, sửa workflow trước khi tăng độ phức tạp model.

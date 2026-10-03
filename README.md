# CIFAR-10 — local development, Kaggle GPU training

Code Python dùng chung ở local và Kaggle. Chỉnh code trên máy cá nhân, gửi snapshot
lên Kaggle để train, sau đó tải artifacts về. Đây là batch job, không phải SSH hay
remote debugger. Python local **3.12+**, quản lý môi trường bằng `uv`.

## Bắt đầu: kết nối và chạy

Chạy các lệnh từ thư mục gốc repository. Chỉ cần thay `YOUR_USERNAME` bằng username Kaggle.

```bash
uv sync --locked
uv tool install kaggle
kaggle auth login

python3 tools/kaggle_workflow.py build --username YOUR_USERNAME
python3 tools/kaggle_workflow.py push
python3 tools/kaggle_workflow.py status
# Khi trạng thái COMPLETE:
python3 tools/kaggle_workflow.py pull
```

Tài khoản Kaggle cần quyền dùng GPU, còn quota và cho phép Internet để tải CIFAR-10.
Thực hiện xác minh tài khoản trên Kaggle nếu được yêu cầu. CLI hỗ trợ credentials
sẵn có ở `~/.kaggle/kaggle.json`; không đưa token vào repo hoặc notebook.
`push` thực sự gửi code và khởi chạy job GPU private. Sau khi gửi thành công,
không cần giữ máy cá nhân mở.

Mặc định: 20 epochs, batch 128, Adam lr 0.001, seed 42, 10% tập train dành cho
validation, AMP khi dùng CUDA. Thay đổi tại [configs/cifar10.toml](configs/cifar10.toml).
Có thể ghi đè cho từng run:

```bash
python3 tools/kaggle_workflow.py build --username YOUR_USERNAME --epochs 5 --lr 0.0005
```

`build` không upload. Snapshot chứa code `.py`, hash SHA-256 và toàn bộ cấu hình đã
resolve. Sửa code/config sau khi build cần build lại. `push`, `status`, `pull` mặc
định dùng **run vừa build**, hoặc chọn `--run-dir build/kaggle/<run-id>`.
Mỗi build tạo kernel riêng. Push lại cùng build tạo phiên bản mới; status/pull lấy
phiên bản mới nhất. Nên build mới cho mỗi thí nghiệm.

GPU mặc định `NvidiaTeslaT4`; đổi bằng `push --accelerator NvidiaL4` nếu tài khoản
hỗ trợ. Training dùng **một GPU**, kể cả khi Kaggle cấp hai T4.

## Cấu trúc và trách nhiệm

```text
configs/cifar10.toml           Cấu hình thí nghiệm, không chứa credentials
src/img_classification/
  config.py                   Đọc/kiểm tra cấu hình, đường dẫn dữ liệu
  dataset.py                  Hợp đồng tensor cache và chuyển ảnh
  prepare_data.py             Tải CIFAR-10 và tạo cache uint8
  models/cnn.py               Nguồn duy nhất của kiến trúc CNN hiện tại
  model.py                    Import tương thích code cũ
  train.py                    Train → validation → best model → test
  artifacts.py                Ghi JSON/checkpoint qua file tạm rồi thay thế
  inspect_forward.py          Quan sát activations phục vụ học tập
  slicing_data.py             Quan sát dữ liệu phục vụ học tập
scripts/                      Wrapper tương thích các lệnh cũ
tools/kaggle_workflow.py      Build/push/status/pull cho hạ tầng Kaggle
tests/                        Kiểm tra offline bằng dữ liệu giả nhỏ
docs/engineering-playbook.md  Quy trình thiết kế áp dụng cho dự án sau
.github/workflows/ci.yml       Chạy test tự động trên GitHub
```

`data/`, `build/`, `outputs/`, `.venv/` không vào Git. `models/resnet.py` là chỗ dự
kiến triển khai, chưa cung cấp ResNet. Kiến trúc CNN đang giữ nguyên để không đổi
baseline cùng lúc refactor; head 32→3→10 có thể hạn chế khả năng học. Muốn cải tiến
accuracy: thực hiện thí nghiệm riêng, so sánh bằng validation.

## Phát triển và kiểm tra local

```bash
uv run python -m unittest discover -s tests -v
uv run img-prepare
uv run img-train --config configs/cifar10.toml --epochs 1 --device cpu
```

Test không tải CIFAR-10, không gọi API Kaggle, không cần GPU. Nó chạy pipeline CPU
với cache giả, kiểm tra artifacts, train/validation không giao nhau, cấu hình,
cache hỏng, snapshot và TOML sinh ra cho runner. Kết quả giả chỉ kiểm chứng luồng
chạy, không đo chất lượng model. CI cần mạng để cài dependency trước khi test.

VS Code chọn `.venv/bin/python`. Dùng entry point hoặc `python -m`; không chạy
trực tiếp file bên trong `src/`. Các wrapper `python -m scripts.train` và
`python -m scripts.prepare_data` vẫn dùng được.

Local mặc định tạo run mới dưới `outputs/local/<timestamp>/`. Có thể đặt
`--output-dir outputs/my-experiment`; nếu đã chứa artifacts training, lệnh dừng
để tránh ghi đè. Đổi data root bằng biến môi trường `IMG_CLASSIFICATION_DATA_DIR`
trước khi chạy, hoặc dùng `img-prepare --rebuild` để tạo lại cache.

## Artifacts và tái lập

Mỗi run hoàn tất có:

| File | Nội dung |
| --- | --- |
| `best.pt` | Weights và optimizer tại epoch validation accuracy tốt nhất |
| `checkpoint.pt` | Weights, optimizer, scaler, epoch mới nhất hoàn tất |
| `config.json` | Cấu hình training thực sự sử dụng |
| `split.json` | Chỉ số train/validation để kiểm tra split |
| `metrics.json` | History, best epoch, test accuracy của best model khi hoàn tất |
| `predictions.png` | Ví dụ dự đoán trên test set |

Run Kaggle có thêm `run.json`, `source.json`, `environment.txt`, `train.log`.
`pull` tải về `outputs/kaggle/<run-id>/`, không tự đợi job hoàn tất.

Validation lấy từ train bằng seed cố định; test chỉ dùng sau chọn best model.
Seed và cuDNN deterministic giúp giảm biến động, không đảm bảo bitwise giống nhau
qua mọi GPU/thư viện. Dataset tải từ `uoft-cs/cifar10` chưa pin revision upstream;
muốn tái lập dữ liệu tuyệt đối cần lưu/version hóa cache hoặc pin revision dataset.

Cache giữ ảnh uint8 `[N,3,32,32]` trong CPU RAM; chuyển float và chia 255 trên device.
AMP chỉ bật khi train CUDA, evaluation dùng float32. Loader dùng 0 worker vì dữ
liệu đã nằm trong RAM; chỉ đổi sau khi đo bottleneck. Không có benchmark GPU nên
chưa khẳng định mức tăng tốc.

Runner giữ nguyên PyTorch/CUDA có sẵn trên Kaggle, cài các thư viện còn thiếu,
ghi lại `pip freeze`. Không chạy `uv sync` trên Kaggle; lockfile chỉ khóa môi trường
local/CI. Môi trường remote có thể khác local. Runner yêu cầu Python 3.12+ và
PyTorch 2.4+ cho API đang sử dụng. Nếu thêm dependency runtime, cập nhật cả
`pyproject.toml` và danh sách dependency trong runner.

Chưa có CLI resume: checkpoint phục vụ khôi phục weights và phát triển resume sau.
Artifacts của job bị dừng cưỡng bức chỉ lấy được nếu Kaggle đã lưu output.
Dữ liệu/cache remote ở thư mục tạm; mỗi job mới chuẩn bị lại dữ liệu.

## Xử lý lỗi thường gặp

- `Kaggle CLI missing`: chạy `uv tool install kaggle`, bảo đảm executable nằm trong PATH.
- Lỗi xác thực/quyền: chạy lại `kaggle auth login`, kiểm tra username và quyền tài khoản.
- `No CUDA GPU allocated`: kiểm tra accelerator, quota, quyền GPU trên Kaggle.
- Download lỗi: kiểm tra Internet trên kernel; xem `train.log` sau khi job có output.
- `Missing tensor cache`: chạy `uv run img-prepare` với cùng data root.
- `Run already exists`: chọn output directory mới, giữ run cũ để đối chiếu.

Tài liệu: [Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
[metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
Bắt đầu xây thói quen với [Engineering playbook](docs/engineering-playbook.md).

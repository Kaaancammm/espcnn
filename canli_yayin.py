import os
import socket
import threading
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DOSYA VE MODEL SEÇİMİ ---
VIDEO_KAYNAGI = "gercek_kolonoskopi_test.mp4"
MODEL_YOLU = "residual_espcn_4x_1k.pth"  # 4 kat büyütme modeli
SCALE = 4

if not os.path.exists(MODEL_YOLU):
    print(f"HATA: '{MODEL_YOLU}' bulunamadı, hiperkvasir modeline dönülüyor.")
    MODEL_YOLU = "residual_espcn_hyperkvasir.pth"
    SCALE = 2

# --- AĞ AYARLARI ---
UDP_IP = "127.0.0.1"
UDP_PORT = 9999
HEDEF_GENISLIK = 640
HEDEF_YUKSEKLIK = 480

# --- RESIDUAL ESPCN MİMARİSİ ---
class ResidualESPCN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
        res = self.relu1(self.conv1(x))
        res = self.relu2(self.conv2(res))
        res = self.pixel_shuffle(self.conv3(res))
        return (base + res).clamp(0.0, 1.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResidualESPCN(scale_factor=SCALE).to(device)
model.load_state_dict(torch.load(MODEL_YOLU, map_location=device))
model.eval()
print(f"YZ Motoru Hazır! Model: {MODEL_YOLU} | Donanım: {device}")

yayin_aktif = True

# --- YAYINCI (SENDER) ---
def yayinci_baslat():
    global yayin_aktif
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(VIDEO_KAYNAGI)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    kare_suresi = 1.0 / fps

    while yayin_aktif:
        t_basla = time.time()
        ret, frame = cap.read()
        if not ret:
            yayin_aktif = False
            break

        # Görüntüyü ağ tasarrufu için küçült
        lr_w = HEDEF_GENISLIK // SCALE
        lr_h = HEDEF_YUKSEKLIK // SCALE
        dusuk_res = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
        _, paket = cv2.imencode(".jpg", dusuk_res, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        veri = paket.tobytes()

        if len(veri) < 65000:
            sock.sendto(veri, (UDP_IP, UDP_PORT))

        gecen = time.time() - t_basla
        if gecen < kare_suresi:
            time.sleep(kare_suresi - gecen)

    cap.release()
    sock.close()

# --- ALICI (RECEIVER) ---
def alici_baslat():
    global yayin_aktif
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(2.0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter('canli_karsilastirma.mp4', fourcc, 25.0, (HEDEF_GENISLIK * 2, HEDEF_YUKSEKLIK))

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    with torch.no_grad():
        while yayin_aktif:
            try:
                veri, _ = sock.recvfrom(65535)
            except socket.timeout:
                break

            np_arr = np.frombuffer(veri, dtype=np.uint8)
            lr_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if lr_bgr is None:
                continue

            # 1. Klasik Bicubic Büyütme
            bic_bgr = cv2.resize(lr_bgr, (HEDEF_GENISLIK, HEDEF_YUKSEKLIK), interpolation=cv2.INTER_CUBIC)

            # 2. ESPCN Model Çıkarımı
            lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
            lr_t = torch.from_numpy(lr_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True) / 255.0

            if device.type == "cuda":
                start_event.record()
                sr_t = model(lr_t)
                end_event.record()
                torch.cuda.synchronize()
                gecikme_ms = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                sr_t = model(lr_t)
                gecikme_ms = (time.perf_counter() - t0) * 1000.0

            sr_rgb = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)

            # Metrik Yazıları
            anlik_fps = 1000.0 / max(gecikme_ms, 0.1)
            cv2.putText(bic_bgr, f"Standart Bicubic | {len(veri)/1024:.1f} KB/kare", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(sr_bgr, f"Residual ESPCN | {gecikme_ms:.1f}ms ({anlik_fps:.0f} FPS)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            karsilastirma = np.hstack((bic_bgr, sr_bgr))
            out_video.write(karsilastirma)

    out_video.release()
    sock.close()
    print("İşlem bitti! 'canli_karsilastirma.mp4' hazırlandı.")

t_yayinci = threading.Thread(target=yayinci_baslat, daemon=True)
t_yayinci.start()
alici_baslat()
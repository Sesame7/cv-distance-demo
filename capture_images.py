import cv2, time, os, sys

# 与主程序保持一致的设置
BACKEND = "msmf"   # "msmf" | "dshow" | "any" | "v4l2"
SRC     = 1
WIDTH   = 2560
HEIGHT  = 1440
PREV_MAX_W = 1280
PREV_MAX_H = 800
FPS_ALPHA  = 0.1
SAVE_DIR   = "./samples"

def die(msg):
    sys.stderr.write(msg.strip()+"\n")
    raise SystemExit(1)

def open_cam(source=0, backend="msmf", width=2560, height=1440):
    backends = {"any":cv2.CAP_ANY, "dshow":cv2.CAP_DSHOW, "msmf":cv2.CAP_MSMF, "v4l2":cv2.CAP_V4L2}
    cap = cv2.VideoCapture(source, backends.get(backend, cv2.CAP_ANY))
    if not cap.isOpened(): die("ERROR: cannot open video source")
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frm = cap.read()
    if not ok or frm is None: die("ERROR: empty frame after open")
    return cap, (frm.shape[1], frm.shape[0])  # (W,H)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    cap, (W_eff, H_eff) = open_cam(SRC, BACKEND, WIDTH, HEIGHT)
    fps_s, t_prev = 0.0, None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # FPS（滑动指数平均）
        t = time.time()
        fps = 0.0 if t_prev is None else (1.0/(t - t_prev) if t > t_prev else 0.0)
        t_prev = t
        fps_s = fps if fps_s == 0.0 else (FPS_ALPHA*fps + (1.0-FPS_ALPHA)*fps_s)

        # 仅用于显示的等比缩放
        h, w = frame.shape[:2]
        scale = min(PREV_MAX_W / w, PREV_MAX_H / h, 1.0)
        disp = cv2.resize(frame, (int(w*scale), int(h*scale))) if scale < 1.0 else frame

        # HUD
        cv2.putText(disp, f"FPS:{int(round(fps_s))}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Preview (s=save, ESC=quit)", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC
            break
        if k == ord('s'):
            fn = f"img_{int(time.time()*1000)}.png"
            cv2.imwrite(os.path.join(SAVE_DIR, fn), frame)
            # 不额外打印；需要的话可短暂闪一下提示框

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

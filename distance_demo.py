import sys, time, cv2
import numpy as np

# ====== 常量（按需修改） ======
CALIB_PATH   = "./calib.npz"   # 标定文件
MARKER_MM    = 47.0            # 方块物理边长（毫米）
CAM_SOURCE   = 1               # 摄像头索引或URL
BACKEND      = "msmf"          # "msmf" | "dshow" | "any"
FRAME_WIDTH  = 2560            # 采集宽
FRAME_HEIGHT = 1440            # 采集高
PREV_MAX_W   = 1280            # 仅用于显示的最大宽
PREV_MAX_H   = 800             # 仅用于显示的最大高
ARUCO_DICT   = "DICT_5X5_50"   # 固定字典
FPS_ALPHA    = 0.1             # FPS 平滑
DIST_ALPHA   = 0.1             # 距离平滑
# ============================

def die(msg: str):
    sys.stderr.write(msg.strip() + "\n")
    sys.exit(1)

def load_calib(path):
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        die(f"ERROR: cannot load calib '{path}': {e}")
    if "K" not in d.files or "dist" not in d.files:
        die("ERROR: calib file missing 'K' or 'dist'")
    K = d["K"].astype(np.float32)
    dist = d["dist"].astype(np.float32).ravel()
    if K.shape != (3,3): die("ERROR: K must be 3x3")
    img_size = tuple(int(x) for x in d["img_size"].ravel()) if "img_size" in d.files else None
    return K, dist, img_size

def scale_K(K, sx, sy):
    K2 = K.copy().astype(np.float32)
    K2[0,0] *= sx; K2[1,1] *= sy
    K2[0,2] *= sx; K2[1,2] *= sy
    return K2

def open_cam(source, backend, w, h):
    backends = {"any":cv2.CAP_ANY, "dshow":cv2.CAP_DSHOW, "msmf":cv2.CAP_MSMF, "v4l2":cv2.CAP_V4L2}
    cap = cv2.VideoCapture(source, backends.get(backend, cv2.CAP_ANY))
    if not cap.isOpened(): die("ERROR: cannot open video source")
    if w: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    if h: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, frm = cap.read()
    if not ok or frm is None: die("ERROR: empty frame after open")
    return cap, (frm.shape[1], frm.shape[0])  # (W,H)

def get_aruco(dict_name):
    if not hasattr(cv2, "aruco"):
        die("ERROR: cv2.aruco not available (pip install opencv-contrib-python)")
    # 修复常量获取：从 cv2.aruco 命名空间拿
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    # 兼容新旧API
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(adict, params)
        return ("new", detector, None, adict)
    except Exception:
        params = cv2.aruco.DetectorParameters_create()
        return ("old", None, params, adict)

def detect_markers(gray, bundle):
    mode, detector, params, adict = bundle
    return detector.detectMarkers(gray) if mode=="new" else \
           cv2.aruco.detectMarkers(gray, adict, parameters=params)

def choose_marker(corners, ids):
    if ids is None or len(corners)==0: return None
    areas = [cv2.contourArea(c.reshape(-1,1,2).astype(np.float32)) for c in corners]
    return corners[int(np.argmax(areas))]

def corners_min_side_px(c4):
    pts = np.asarray(c4, np.float32).reshape(-1,2)[:4]
    ds = [np.linalg.norm(pts[(i+1)%4]-pts[i]) for i in range(4)]
    return float(min(ds))

def robust_pnp(c4, K, dist, Lm, D_hint):
    import math
    def finite_and_reasonable(vec, max_abs):
        v = np.asarray(vec, dtype=np.float64).ravel()
        return np.isfinite(v).all() and (np.max(np.abs(v)) < max_abs)

    c4 = np.asarray(c4, np.float32).reshape(-1,2)[:4]
    h = Lm/2.0
    obj = np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]], np.float32)
    obj_ = obj.reshape(-1,1,3); img_ = c4.reshape(-1,1,2)

    flags = []
    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE"): flags.append(cv2.SOLVEPNP_IPPE_SQUARE)
    if hasattr(cv2, "SOLVEPNP_SQPNP"):       flags.append(cv2.SOLVEPNP_SQPNP)
    flags += [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP]

    if D_hint is None or not np.isfinite(D_hint) or D_hint <= 1e-6:
        D_hint = 1.0

    best = None
    for flag in flags:
        ok, rvec, tvec = cv2.solvePnP(obj_, img_, K, dist, flags=flag)
        if not ok: continue
        if (not finite_and_reasonable(rvec, 1e3)) or (not finite_and_reasonable(tvec, 100.0)):
            continue
        rvec = np.asarray(rvec, np.float64).reshape(3)
        tvec = np.asarray(tvec, np.float64).reshape(3)
        proj,_ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        diff = proj.reshape(-1,2).astype(np.float64) - c4.astype(np.float64)
        reproj = float(cv2.norm(diff, cv2.NORM_L2)) / 4.0
        D = float(np.linalg.norm(tvec))
        rel = abs(math.log(max(D,1e-9) / max(float(D_hint),1e-9)))
        score = reproj + 1000.0*rel
        if (best is None) or (score < best[0]): best = (score, rvec, tvec)
    if best is None: raise RuntimeError("PnP failed (all candidates rejected)")
    return best[1], best[2]

def corner_dists_and_cam(rvec, tvec, Lm):
    h = Lm/2.0
    obj = np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]], np.float32)
    R,_ = cv2.Rodrigues(rvec)
    cam_pts = (R @ obj.T).T + tvec.reshape(1,3)
    dists = [float(np.linalg.norm(v)) for v in cam_pts]
    return dists, cam_pts

def draw_column_right(disp, lines, color=(255,0,0)):
    pad=10; font=cv2.FONT_HERSHEY_SIMPLEX; scale=0.55; thick=1; gap=18
    maxw=max((cv2.getTextSize(s,font,scale,thick)[0][0] for s in lines), default=0)
    H,W=disp.shape[:2]; x=W-maxw-pad; y=40
    for i,s in enumerate(lines):
        cv2.putText(disp, s, (x, y+i*gap), font, scale, color, thick, cv2.LINE_AA)

def main():
    # 基础检查
    if MARKER_MM is None or MARKER_MM <= 0: die("ERROR: invalid MARKER_MM")
    Lm = float(MARKER_MM)/1000.0

    # 相机
    backend_map={"any":cv2.CAP_ANY,"dshow":cv2.CAP_DSHOW,"msmf":cv2.CAP_MSMF,"v4l2":cv2.CAP_V4L2}
    cap = cv2.VideoCapture(CAM_SOURCE, backend_map.get(BACKEND, cv2.CAP_ANY))
    if not cap.isOpened(): die("ERROR: cannot open video source")
    if FRAME_WIDTH:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    if FRAME_HEIGHT: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    ok, frame = cap.read()
    if not ok or frame is None: die("ERROR: empty frame after open")
    W,H = frame.shape[1], frame.shape[0]

    # 标定与 K 缩放
    K_raw, dist, calib_size = load_calib(CALIB_PATH)
    K = scale_K(K_raw, W/float(calib_size[0]), H/float(calib_size[1])) if (calib_size and (W,H)!=tuple(calib_size)) else K_raw

    # ArUco
    if not hasattr(cv2, "aruco"): die("ERROR: cv2.aruco not available")
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    try:
        params = cv2.aruco.DetectorParameters(); detector = cv2.aruco.ArucoDetector(adict, params)
        use_new=True
    except Exception:
        params = cv2.aruco.DetectorParameters_create(); use_new=False

    fps_s=0.0; D_s=None; t_prev=None
    names=["TL","TR","BR","BL"]

    while True:
        ok, frame = cap.read()
        if not ok or frame is None: continue

        # FPS
        t=time.time(); fps=0.0 if t_prev is None else (1.0/(t-t_prev) if t>t_prev else 0.0); t_prev=t
        fps_s = fps if fps_s==0.0 else (FPS_ALPHA*fps + (1.0-FPS_ALPHA)*fps_s)

        # 检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if use_new: corners, ids, _ = detector.detectMarkers(gray)
        else:       corners, ids, _ = cv2.aruco.detectMarkers(gray, adict, parameters=params)
        c = choose_marker(corners, ids)
        D_text="--"

        # 显示缩放
        scale = min(PREV_MAX_W/W, PREV_MAX_H/H, 1.0)

        lines=[]
        if c is not None:
            # 框
            pts = np.asarray(c, np.float32).reshape(-1,2)[:4]
            for i in range(4):
                p1 = tuple(pts[i].astype(int)); p2 = tuple(pts[(i+1)%4].astype(int))
                cv2.line(frame, p1, p2, (0,255,0), 2)

            # 百分比（相对原始宽高）
            uvp = [ (float(u)/W*100.0, float(v)/H*100.0) for (u,v) in pts ]

            # 先验
            s_px = corners_min_side_px(c); fx=float(K[0,0])
            D_hint = (fx*Lm/s_px) if s_px>0 else None

            try:
                rvec, tvec = robust_pnp(c, K, dist, Lm, D_hint)
                dists, cam_pts = corner_dists_and_cam(rvec, tvec, Lm)
                D_med = float(np.median(dists))
                D_s = D_med if D_s is None else (DIST_ALPHA*D_med + (1.0-DIST_ALPHA)*D_s)
                D_text = f"{D_s:.2f} m"

                # 右侧单列（蓝色）
                for name, (x_pct, y_pct), (_,_,Z) in zip(names, uvp, cam_pts):
                    lines.append(f"{name}: x:{x_pct:.1f}%  y:{y_pct:.1f}%  Z:{Z:.2f} m")
            except Exception:
                pass

        # 显示帧
        disp = cv2.resize(frame, (int(W*scale), int(H*scale))) if scale<1.0 else frame

        # 左上角：第一行 D，第二行 FPS（绿色）
        cv2.putText(disp, f"D = {D_text}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(disp, f"FPS:{int(round(fps_s))}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # 右侧蓝色单列
        if lines:
            draw_column_right(disp, lines, color=(255,0,0))  # 蓝色

        cv2.imshow("Aruco Distance (ESC=quit)", disp)
        if (cv2.waitKey(1) & 0xFF) == 27: break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

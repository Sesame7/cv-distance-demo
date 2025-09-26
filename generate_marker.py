import cv2, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="DICT_5X5_50")
    ap.add_argument("--id", type=int, default=0)
    ap.add_argument("--size", type=int, default=800, help="marker side pixels")
    ap.add_argument("--border", type=int, default=1, help="quiet zone (bits)")
    ap.add_argument("--out", default="aruco_id0.png")
    args = ap.parse_args()

    if not hasattr(cv2, "aruco"):
        raise SystemExit("ERROR: opencv-contrib-python required")

    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    img = cv2.aruco.drawMarker(adict, args.id, args.size)
    # 可选：加白边（quiet zone），以免边缘干扰
    img = cv2.copyMakeBorder(img, args.border*10, args.border*10, args.border*10, args.border*10,
                             cv2.BORDER_CONSTANT, value=255)

    cv2.imwrite(args.out, img)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()

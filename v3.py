from cvzone.HandTrackingModule import HandDetector
import cv2
import pyautogui
import time

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 设置宽度
cap.set(4, 480)  # 设置高度

# 初始化手势检测器
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# 用于记录上一帧手的中心位置和时间
prev_center = None
start_move_time = None

# 上次触发时间
last_trigger_time = 0

# 用于手势稳定性判断（连续帧数）
stable_frame_count = 0
required_stable_frames = 3  # 需要连续3帧为五指张开才开始检测移动

# 平滑参数（指数加权平均）
alpha = 0.6  # 越小越平滑
smooth_center = (0, 0)

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand = hands[0]
        center = hand['center']
        fingers = detector.fingersUp(hand)

        current_time = time.time()

        # 对 center 做平滑处理
        if prev_center is None:
            smooth_center = center
        else:
            smooth_center = (
                int(alpha * center[0] + (1 - alpha) * prev_center[0]),
                int(alpha * center[1] + (1 - alpha) * prev_center[1])
            )

        # 条件1：必须是五指张开 + 手心朝前
        if fingers == [1, 1, 1, 1, 1] and hand["type"] == "Right":
            stable_frame_count += 1
        else:
            stable_frame_count = 0

        # 如果连续几帧都是五指张开 + 手心朝前，则开始检测移动
        if stable_frame_count >= required_stable_frames:
            if prev_center is not None and start_move_time is not None:
                dx = smooth_center[0] - prev_center[0]
                dy = smooth_center[1] - prev_center[1]

                if abs(dx) > 200 and abs(dy) < 150:
                    duration = current_time - start_move_time
                    if duration < 1.0:  # 在1秒内完成
                        if current_time - last_trigger_time > 1.0:
                            pyautogui.press('space')
                            print("空格键已触发")
                            last_trigger_time = current_time  # 更新上次触发时间

            # 更新起点信息
            prev_center = smooth_center
            start_move_time = current_time

        else:
            # 如果不是五指或不是手心朝前，重置移动起点
            prev_center = smooth_center
            start_move_time = current_time

        # 打印手指数量
        print(f'[DEBUG] H1 = {fingers.count(1)}, Type: {hand["type"]}, dx={dx}, dy={dy}, duration={duration:.2f}')

        if len(hands) == 2:
            hand2 = hands[1]
            fingers2 = detector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

        print("")  # 换行

    else:
        # 没有检测到手时，重置移动起点和稳定计数
        prev_center = None
        start_move_time = None
        stable_frame_count = 0
        smooth_center = (0, 0)

    # 显示图像
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break  # 按 Q 键退出程序

# 释放资源
cap.release()
cv2.destroyAllWindows()

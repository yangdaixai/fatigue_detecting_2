
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math

 



# 播放语音

import pygame
def voice_prompt(mp3Path):
    print("播放音乐：", mp3Path)

# def voice_prompt(mp3Path):
#     pygame.mixer.init()  # 初始化音频引擎
#     #pygame.mixer.music.load('./MP3/闭眼开车.MP3')  # 加载音频文件
#     pygame.mixer.music.load(mp3Path)  # 加载音频文件
#     pygame.mixer.music.play()  # 播放音频

#     while pygame.mixer.music.get_busy():  # 检查音乐是否仍在播放
#         pygame.time.Clock().tick(10)  # 等待播放完成


# 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  #33左眉左上角
                         [1.330353, 7.122144, 6.903745],  #29左眉右角
                         [-1.330353, 7.122144, 6.903745], #34右眉左角
                         [-6.825897, 6.760612, 4.402142], #38右眉右上角
                         [5.311432, 5.485328, 3.987654],  #13左眼左上角
                         [1.789930, 5.393625, 4.413414],  #17左眼右上角
                         [-1.789930, 5.393625, 4.413414], #25右眼左上角
                         [-5.311432, 5.485328, 3.987654], #21右眼右上角
                         [2.005628, 1.409845, 6.165652],  #55鼻子左上角
                         [-2.005628, 1.409845, 6.165652], #49鼻子右上角
                         [2.774015, -2.080775, 5.048531], #43嘴左上角
                         [-2.774015, -2.080775, 5.048531],#39嘴右上角
                         [0.000000, -3.116408, 6.097667], #45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])#6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]# 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

def get_head_pose(shape):# 头部姿态估计
    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))# 以8行2列显示

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)#罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))# 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    # euler_angle 元组中：
    # 俯仰角（Pitch）：绕 X 轴的旋转，表示物体向上或向下的倾斜程度，表示头部的上下点头动作。
    # 偏航角（Yaw）：绕 Y 轴的旋转，表示物体左右转向的角度，表示头部的左右转动。
    # 滚转角（Roll）：绕 Z 轴的旋转，表示物体绕前进方向旋转的角度，表示头部的左右摇头动作。

    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    # pitch, yaw, roll = euler_angle
    print("euler_angle:",euler_angle)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

 
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    return reprojectdst, euler_angle# 投影误差，欧拉角

def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])# 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear
 
def mouth_aspect_ratio(mouth):# 嘴部
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# 定义常数
# 眼睛长宽比
# 闪烁阈值
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 5
# 打哈欠长宽比
# 闪烁阈值
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
# 瞌睡点头
HAR_THRESH = 10
HAR_LEFT_RIGHT_THRESH = 25
NOD_AR_CONSEC_FRAMES = 3
# 初始化帧计数器和眨眼总数
COUNTER = 0
TOTAL = 0
# 初始化帧计数器和打哈欠总数
mCOUNTER = 0
mTOTAL = 0
# 初始化帧计数器和点头总数
hCOUNTER = 0
hTOTAL = 0

# 初始化帧计数器和左右转头总数
LeftRightCounter = 0

# 连续结果相同帧数
CONTINUOUS_FRAMES =60

# 是否疲劳驾驶标识

# 一直闭眼
CLOSE_EYE_FATIGUED_DRIVING = False
#一直低头
UP_DOWN_FATIGUED_DRIVING = False
#一直左/右转脸不看前方
LEFT_RIGHT_FATIGUED_DRIVING = False


FATIGUED_DRIVING = False
# 初始化DLIB的人脸检测器（HOG），然后创建面部标志物预测
print("[INFO] loading facial landmark predictor...")
# 第一步：使用dlib.get_frontal_face_detector() 获得脸部位置检测器
detector = dlib.get_frontal_face_detector()
# 第二步：使用dlib.shape_predictor获得脸部特征位置检测器
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
 
# 第三步：分别获取左右眼面部标志的索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# 第四步：打开cv2 本地摄像头
cap = cv2.VideoCapture(0)
# 设置视频编码器并创建VideoWriter对象  
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # 设置视频编码器  

ret, frame1 = cap.read()
frame1 = imutils.resize(frame1, width=720)
#frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
# 获取图片的高度、宽度和通道数  
height, width, channels = frame1.shape 
out = cv2.VideoWriter('WIN_OUT.mp4', fourcc, 30.0, (width, height))  # 创建VideoWriter对象
# 从视频流循环帧
while True:
    # 第五步：进行循环，读取图片，并对图片做维度扩大，并进灰度化
    ret, frame = cap.read()
    if not ret:  
        print("去取图片为空")
        break  
    frame = imutils.resize(frame, width=720)
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 第六步：使用detector(gray, 0) 进行脸部位置检测
    rects = detector(gray, 0)
    
    # 第七步：循环脸部位置信息，使用predictor(gray, rect)获得脸部特征位置的信息
    for rect in rects:
        shape = predictor(gray, rect)
        
        # 第八步：将脸部特征信息转换为数组array的格式
        shape = face_utils.shape_to_np(shape)
        
        # 第九步：提取左眼和右眼坐标
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        # 嘴巴坐标
        mouth = shape[mStart:mEnd]        
        
        # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        # 打哈欠
        mar = mouth_aspect_ratio(mouth)
 
        # 第十一步：使用cv2.convexHull获得凸包位置，使用drawContours画出轮廓位置进行画图操作
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
 
        # 第十二步：进行画图操作，用矩形框标注人脸
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)    
 
        '''
            分别计算左眼和右眼的评分求平均作为最终的评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示进行了一次眨眼活动
        '''
        # 第十三步：循环，满足条件的，眨眼次数+1
        if ear < EYE_AR_THRESH:# 眼睛长宽比：0.2
            COUNTER += 1
           
        else:
            # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
            if COUNTER >= EYE_AR_CONSEC_FRAMES:# 阈值：3
                TOTAL += 1
            # 重置眼帧计数器
            COUNTER = 0
        

        # 如果连续数帧，都是闭眼则直接提醒疲劳驾驶
        if COUNTER > CONTINUOUS_FRAMES:
            CLOSE_EYE_FATIGUED_DRIVING = True
            COUNTER = 0

        # 第十四步：进行画图操作，同时使用cv2.putText将眨眼次数进行显示
        #cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)     
        #cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, "Blinks: {}".format(TOTAL), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
        '''
            计算张嘴评分，如果小于阈值，则加1，如果连续3次都小于阈值，则表示打了一次哈欠，同一次哈欠大约在3帧
        '''
        # 同理，判断是否打哈欠    
        if mar > MAR_THRESH:# 张嘴阈值0.5
            mCOUNTER += 1
            #cv2.putText(frame, "Yawning!", (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # 如果连续3次都小于阈值，则表示打了一次哈欠
            if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:# 阈值：3
                mTOTAL += 1
            # 重置嘴帧计数器
            mCOUNTER = 0



        #cv2.putText(frame, "COUNTER: {}".format(mCOUNTER), (150, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, "Yawning: {}".format(mTOTAL), (450, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        """
        瞌睡点头
        """
        # 第十五步：获取头部姿态
        reprojectdst, euler_angle = get_head_pose(shape)

        
        # euler_angle 元组中：
        # 俯仰角（Pitch）：绕 X 轴的旋转，表示物体向上或向下的倾斜程度，表示头部的上下点头动作，低头正角度
        # 偏航角（Yaw）：绕 Y 轴的旋转，表示物体左右转向的角度，表示头部的左右转动。右转是正角度
        # 滚转角（Roll）：绕 Z 轴的旋转，表示物体绕前进方向旋转的角度，表示头部的左右摇头动作。右摇头是正角度
        #pitch, yaw, roll = euler_angle
        pitch = euler_angle[0, 0]
        yaw = euler_angle[1, 0]
        roll = euler_angle[2, 0]
        # 转换弧度到度
        # pitch_degrees = math.degrees(euler_angle[0, 0])
        # yaw_degrees = math.degrees(euler_angle[1, 0])
        # roll_degrees = math.degrees(euler_angle[2, 0])
        if(pitch > 0):
            print("低头pitch上下点头角度:", euler_angle[0, 0])
        else:
            print("抬头pitch上下点头角度:", euler_angle[0, 0])
        if(yaw > 0):
            print("右转yaw上下点头角度:", euler_angle[1, 0])
        else:
            print("左转yaw上下点头角度:", euler_angle[1, 0])
        if(roll > 0):
            print("右摇头roll上下点头角度:", euler_angle[2, 0])
        else:
            print("左摇头roll上下点头角度:", euler_angle[2, 0])

        # 低头判断
        if pitch > HAR_THRESH:# 点头阈值
            hCOUNTER += 1
        else:
            # 如果连续3次都小于阈值，则表示瞌睡点头一次
            if hCOUNTER >= NOD_AR_CONSEC_FRAMES:# 阈值：3
                hTOTAL += 1
            # 重置点头帧计数器
            hCOUNTER = 0
         # 如果连续数帧，都是打哈欠则直接提醒疲劳驾驶
        if hCOUNTER > CONTINUOUS_FRAMES:
        
            UP_DOWN_FATIGUED_DRIVING = True
            hCOUNTER = 0
            hTOTAL = 0


        # 左转或者右转头判断
        if abs(yaw) > HAR_LEFT_RIGHT_THRESH:# 转头阈值
            LeftRightCounter += 1
         # 如果连续数帧，都是左转或者右转直接提醒不集中注意力驾驶
        if LeftRightCounter > CONTINUOUS_FRAMES:
            LEFT_RIGHT_FATIGUED_DRIVING = True
            LeftRightCounter = 0

        # 绘制正方体12轴
        for start, end in line_pairs:
            #cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))
            start_point = reprojectdst[start]
            end_point = reprojectdst[end]
    
            # 确保坐标是整数
            start_point = (int(start_point[0]), int(start_point[1]))
            end_point = (int(end_point[0]), int(end_point[1]))
    
            # 画线
            cv2.line(frame, start_point, end_point, (0, 0, 255))

        # 显示角度结果
        # 俯仰角（Pitch）：绕 X 轴的旋转，表示物体向上或向下的倾斜程度，表示头部的上下点头动作。
        # 偏航角（Yaw）：绕 Y 轴的旋转，表示物体左右转向的角度，表示头部的左右转动。
        # 滚转角（Roll）：绕 Z 轴的旋转，表示物体绕前进方向旋转的角度，表示头部的左右摇头动作。
        cv2.putText(frame, "Pitch(X): " + "{:7.2f}".format(euler_angle[0, 0]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), thickness=2)# GREEN
        cv2.putText(frame, "Yaw(Y): " + "{:7.2f}".format(euler_angle[1, 0]), (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.75, (255, 0, 0), thickness=2)# BLUE
        cv2.putText(frame, "Roll(Z): " + "{:7.2f}".format(euler_angle[2, 0]), (10, 120), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), thickness=2)# RED    
       # cv2.putText(frame, "Nod: {}".format(hTOTAL), (450, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        
            
        # 第十六步：进行画图操作，68个特征点标识
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        print('嘴巴实时长宽比:{:.2f} '.format(mar)+"\t是否张嘴："+str([False,True][mar > MAR_THRESH]))
        print('眼睛实时长宽比:{:.2f} '.format(ear)+"\t是否眨眼："+str([False,True][COUNTER>=1]))
    

    # 高危险程度警告
    # 一直闭眼
    if CLOSE_EYE_FATIGUED_DRIVING :
        CLOSE_EYE_FATIGUED_DRIVING =False
        print("一直闭眼")
        voice_prompt('./MP3/开车一直闭眼.MP3')
    elif UP_DOWN_FATIGUED_DRIVING : # 一直低头
        UP_DOWN_FATIGUED_DRIVING =False
        print("一直低头")
        voice_prompt('./MP3/开车一直低头.MP3')
    elif LEFT_RIGHT_FATIGUED_DRIVING : # 一直左看右看
        LEFT_RIGHT_FATIGUED_DRIVING =False
        print("一直左看右看")
        voice_prompt('./MP3/开车一直左看或者右看.MP3')
    else:
        # 低危险程度警告
        # 确定疲劳提示:眨眼50次，打哈欠15次，瞌睡点头15次

        if(hTOTAL >= 15):
            hTOTAL = 0
            print("频繁点头")
            voice_prompt('./MP3/频繁低头开车.MP3')
        elif(TOTAL >= 50):
            TOTAL = 0
            cv2.putText(frame, "频繁眨眼!!!", (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            print("频繁眨眼")
            voice_prompt('./MP3/开车频繁眨眼.MP3')
            
        elif(mTOTAL >= 15):
            mTOTAL= 0
            print("打哈欠")
            voice_prompt('./MP3/打哈欠开车.MP3')
       

    # 按q退出
    #cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # 将处理后的frame写入到视频文件中  
    out.write(frame)  
    # 窗口显示 show with opencv
    cv2.imshow("Frame", frame)
    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# 释放摄像头 release camera
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()


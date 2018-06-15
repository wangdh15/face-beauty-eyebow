import cv2
import dlib
import numpy as np 
import sys
from PIL import Image

# 创建一个人脸检测器类。
face_detector = dlib.get_frontal_face_detector()

# 官方提供的模型构建特征提取器，提前训练好的，可以直接进行人脸关键点的检测。
# 18-22为左眉毛      23-27为右眉毛  37-40为左眼皮    43-46为右眼皮

landmark_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def process(img_path):
	global face_detector, landmark_predictor
	img = cv2.imread( img_path )
	left = cv2.imread('./images/11_left.jpeg')
	right = cv2.imread('./images/11_right.jpeg')

	img_processed = img.copy()

	# 使用检测器对输入的图片进行人脸检测,返回矩形坐标
	facesrect = face_detector(img)

	left_eye_bow = []
	left_eye = []
	right_eye_bow = []
	right_eye = []
	x_left_sum = 0
	y_left_sum = 0
	x_right_sum = 0
	y_right_sum = 0
	y_left_sum_1 = 0
	y_right_sum_1 = 0

	for k, d in enumerate(facesrect):
		shape = landmark_predictor(img, d)
		for i in range(17, 22):
			# 提取左眼眉毛的特征点。由于效果原因，眉毛上面的特征点向上取20个像素。
			left_eye_bow.append((shape.part(i).x, shape.part(i).y - 20))
			x_left_sum += shape.part(i).x
			y_left_sum += shape.part(i).y 
			y_left_sum_1 += shape.part(i).y

		for i in range(22, 27):
			# 提取右眼眉毛的特征点。由于效果原因，眉毛上面的特征点向上取20个像素。
			right_eye_bow.append((shape.part(i).x, shape.part(i).y - 20))
			# 将标记点的具体坐标保存下来，用于计算后面的变换矩阵。
			x_right_sum += shape.part(i).x
			y_right_sum += shape.part(i).y 
			y_right_sum_1 += shape.part(i).y

		flag = 0 # 用于标记正在存的是眼睛下面的哪个特征点。如果是最外侧的点，就用上面最外测的点来做平均
		for i in range(36,  40):
			if flag <= 2:
				left_eye.append((int(shape.part(i).x * 0.6 + left_eye_bow[flag][0] * 0.4), int(shape.part(i).y * 0.6 + left_eye_bow[flag][1] * 0.4)))
			else:
				left_eye.append((int(shape.part(i).x * 0.6 + left_eye_bow[flag+1][0] * 0.4), int(shape.part(i).y * 0.6 + left_eye_bow[flag+1][1] * 0.4)))
			# 继续求和，用于后面求中心点
			x_left_sum += left_eye[flag][0]
			y_left_sum += left_eye[flag][1]
			y_left_sum_1 += left_eye[flag][1]
			flag += 1

		flag = 0 # 用于标记正在存的是眼睛下面的哪个特征点。如果是最外侧的点，就用上面最外测的点来做平均
		for i in range(42, 46):
			if flag <= 2:
				right_eye.append((int(shape.part(i).x * 0.6 + right_eye_bow[flag][0] * 0.4), int(shape.part(i).y * 0.6 + right_eye_bow[flag][1] * 0.4)))
			else:
				right_eye.append((int(shape.part(i).x * 0.6 + right_eye_bow[flag+1][0] * 0.4), int(shape.part(i).y * 0.6 + right_eye_bow[flag+1][1] * 0.4)))
			# 继续求和，用于后面求中心点
			x_right_sum += right_eye[flag][0]
			y_right_sum += right_eye[flag][1]
			y_right_sum_1 += right_eye[flag][1]
			flag += 1

	# 分别求出左右眼去除眉毛的中心点
	center_left = (int(x_left_sum/9), int(y_left_sum/9))
	center_right = (int(x_right_sum/9), int(y_right_sum/9))

	# 求出两眼用于贴上眉毛的中心点
	center_left_merge = (int(x_left_sum/9), int(y_left_sum_1/9))
	center_right_merge = (int(x_right_sum/9), int(y_right_sum_1/9))

	# 将左右眼的眼睛特征点复制一份
	left_eye_copy = left_eye.copy()
	right_eye_copy = right_eye.copy()

	# 将左右眉毛的特征点复制一份
	left_eye_bow_copy = left_eye_bow.copy()
	right_eye_bow_copy = right_eye_bow.copy()

	# 转化为numpy的array类型，复制的部分y坐标加10,用于平衡位置。
	left_eye_bow_copy = np.array(left_eye_bow_copy)
	left_eye_bow_copy = left_eye_bow_copy + np.array([0, 10])
	right_eye_bow_copy = np.array(right_eye_bow_copy)
	right_eye_bow_copy = right_eye_bow_copy + np.array([0, 10])

	# 对眼睛的特征点顺序取反，使得其能够和眉毛的特征点围成一个凸多边形
	left_eye_copy.reverse()
	right_eye_copy.reverse()

	# 将眉毛的特征点和眼睛的特征点拼在一起
	ploy_left = left_eye_bow + left_eye_copy
	ploy_right = right_eye_bow + right_eye_copy

	# 将数据类型转化为array类型
	left_eye_bow = np.array(left_eye_bow)
	left_eye = np.array(left_eye)
	right_eye_bow = np.array(right_eye_bow)
	right_eye = np.array(right_eye)


	ploy_left = np.array(ploy_left)
	ploy_right = np.array(ploy_right)

	# 初始化左右眼的位置mask以及皮肤mask
	mask_left = np.zeros(img.shape, img.dtype)
	skin_mask_left = np.zeros(img.shape, img.dtype)
	mask_right = np.zeros(img.shape, img.dtype)
	skin_mask_right = np.zeros(img.shape, img.dtype)

	# 取出眉毛特征点上面10个像素点的值作为皮肤值
	skin_color_left = img[left_eye_bow[0, 0], left_eye_bow[0, 1] - 10]
	skin_color_right = img[right_eye_bow[0, 0], right_eye_bow[0, 1] - 10]

	# 使用fillPoly函数对mask进行填充
	cv2.fillPoly(mask_left, [ploy_left], (255, 255, 255))
	cv2.fillPoly(skin_mask_left, [ploy_left], skin_color_left.tolist())
	cv2.fillPoly(mask_right, [ploy_right], (255, 255, 255))
	cv2.fillPoly(skin_mask_right, [ploy_right], skin_color_right.tolist())

	# 使用possion融合的方法去除眉毛
	output_remove_left_eyebow = cv2.seamlessClone(skin_mask_left, img, mask_left, center_left, cv2.NORMAL_CLONE)
	output_remove_right_eyebow = cv2.seamlessClone(skin_mask_right, output_remove_left_eyebow, mask_right, center_right, cv2.NORMAL_CLONE)

	# 将去除眉毛的照片输出为remove_eyebow.png
	cv2.imwrite("remove_eyebow.png",output_remove_right_eyebow)
	cv2.imwrite("./mask_right.png" , mask_right)
	# cv2.imwrite("./cut_eyebow.png" , output)

	# 保存我们自己裁剪的眉毛的关键点的坐标，只是眉毛的坐标，目的是后面作变换使用
	left_points = np.array([(8, 27), (22, 14), (58, 9), (87, 15), (108, 23)])
	right_points = np.array([(4, 26), (27, 15), (50, 9), (77, 14), (92, 27)])

	# 初始化要填充眉毛使用的mask
	mask_left = np.zeros(left.shape, left.dtype)
	mask_right = np.zeros(right.shape, right.dtype)

	# 自己裁剪的所有关键点的坐标，包含眼睛的关键点，为了后面填充眉毛求mask所用
	ploy_left = np.array([(8, 27), (22, 14), (58, 9), (87, 15), (108, 23), (96, 40), (67, 29), (38, 26), (18, 37)])
	ploy_right = np.array([(4, 26), (27, 15), (50, 9), (77, 14), (92, 27), (88, 32), (59, 27), (30, 36), (7, 39)])

	# 填充mask
	cv2.fillPoly(mask_left, [ploy_left], (255, 255, 255))
	cv2.fillPoly(mask_right, [ploy_right], (255, 255, 255))

	# 对裁剪的眉毛求变换矩阵
	H_left, mask = cv2.findHomography(left_points, left_eye_bow_copy, cv2.RANSAC)
	H_right, mask = cv2.findHomography(right_points, right_eye_bow_copy, cv2.RANSAC)

	# 将变换矩阵作用在眉毛以及mask上
	left_out = cv2.warpPerspective(left, H_left, (img.shape[1], img.shape[0]))
	mask_left_out = cv2.warpPerspective(mask_left, H_left, (img.shape[1], img.shape[0]))
	right_out = cv2.warpPerspective(right, H_right, (img.shape[1], img.shape[0]))
	mask_right_out = cv2.warpPerspective(mask_right, H_right, (img.shape[1], img.shape[0]))

	# 将变换完的眉毛以及mask在去除眉毛的图片上面进行possion融合。
	output_left = cv2.seamlessClone((1 * left_out).astype(np.uint8), output_remove_right_eyebow, mask_left_out, center_left_merge, cv2.MONOCHROME_TRANSFER)
	output_right = cv2.seamlessClone((0.5 * right_out).astype(np.uint8), output_left, mask_right_out, center_right_merge, cv2.NORMAL_CLONE)

	# 输出图片。
	
	name = img_path.split('/')[-1].split('.')[0]
	save_path = './images/'+ name + '_result' + '.jpg'
	cv2.imwrite(save_path , output_right)
	temple = Image.open(save_path)
	save_path = './images/'+ name + '_result' + '.gif'
	temple.save(save_path , 'gif')



## 前面为函数部分  后面为GUI



import tkinter as tk

def process1():
    global img, label1, window, img_path, img_out, label2, img_path_out
    temple = img_path.get()
    name = temple.split('/')[-1].split('.')[0]
    final_path = './images/'+name+'.jpg'
    process(final_path)
    img = tk.PhotoImage(file=img_path.get())
    label1 = tk.Label(window, image=img)
    label1.place(x=0, y=50)
    save_path = './images/'+ name + '_result' + '.gif'
    img_out = tk.PhotoImage(file=save_path)
    label2 = tk.Label(window, image=img_out)
    label2.place(x=500, y=50)

window = tk.Tk()
window.title('人脸美妆修眉')
window.geometry('2000x1000')

label_img_path = tk.Label(window, text='Image Path:', font=('Helvetica', '14', 'bold'))
label_img_path.place(x=50, y=900)

img_path = tk.StringVar()
img_path.set('./images/welcome.gif')
img_path_out = tk.StringVar()
img_path_out.set('./images/welcome.gif')


l1 = tk.Label(window, text='Original Image： ',font=('Helvetica', '14', 'bold'))
l1.place(x=0, y=0)

l2 = tk.Label(window, text='Processed Image： ',font=('Helvetica', '14', 'bold'))
l2.place(x=500, y=0)


entry_img_path = tk.Entry(window, textvariable=img_path)
entry_img_path.place(x=160, y=905)

img = tk.PhotoImage(file=img_path.get())
label1 = tk.Label(window, image=img)
label1.place(x=0, y=50)

img_out = tk.PhotoImage(file=img_path_out.get())
label2 = tk.Label(window, image=img_out)
label2.place(x=500, y=50)

label_img_path = tk.Label(window, text='Image Path:', font=('Helvetica', '14', 'bold'))
label_img_path.place(x=50, y=900)

btn = tk.Button(window, text='修眉', command=process1)
btn.place(x=500, y=905)

window.mainloop()

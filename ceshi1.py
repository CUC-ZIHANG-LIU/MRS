import tensorflow
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os
import pickle
import pandas as pd
import numpy as np
import webbrowser
import warnings
import tensorflow_text as text
import tensorflow as tf
import tkinter as tk
import pickle
from tkinter import IntVar, filedialog
import tensorflow_text as text

from datetime import datetime

warnings.filterwarnings("ignore")


class EmotionDetection:
  
    def __init__(self, ):

       

        #self.emotion_labels = ['sad', 'neutral', 'happy', 'angry']
        '''
        with open('vgg_model.pkl', 'rb') as f:
            self. model= pickle.load(f)
        '''
        #     #print(f)
        # file = open('cnn_model_1.pkl')
        '''
        tf.saved_model.LoadOptions(
            allow_partial_checkpoint=True,
            experimental_io_device=None,
            experimental_skip_checkpoint=False
        )
        '''
        #self.model = tensorflow.keras.models.load_model("cnn2.h5")  # 用这种保存和导入方式，就不受设备影响了
        # localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="job:localhost")
        # self.model = tf.keras.models.load_model('convLstm_model.pkl', options=localhost_save_option)
        # self.model = pd.read_pickle('./cnn_model.pkl')
        # f = filedialog.askopenfilename()
        # self.model = pickle.load(f)
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        self.root = tk.Tk()
        self.root.title("音乐推荐系统")

        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root)
        self.panel.pack(padx=10, pady=10)

 
        self.lbE=tk.Label(self.root,text='用户情绪:')
        self.lbE.pack()
        self.lbEC=tk.Label(self.root,text='[]')
        self.lbEC.pack()

        self.lbL=tk.Label(self.root,text='用户地点:')
        self.lbL.pack()
        self.var = IntVar()
        self.rd1 = tk.Radiobutton(self.root,text="卧室",variable=self.var,value=0,command=self.Mysel)
        self.rd1.pack()
        self.rd2 = tk.Radiobutton(self.root,text="客厅",variable=self.var,value=1,command=self.Mysel)
        self.rd2.pack()
        self.rd3 = tk.Radiobutton(self.root,text="室内办公地点",variable=self.var,value=2,command=self.Mysel)
        self.rd3.pack()
        self.rd4 = tk.Radiobutton(self.root,text="浴室",variable=self.var,value=3,command=self.Mysel)
        self.rd4.pack()

        self.lbT=tk.Label(self.root,text='当前时间:')
        self.lbT.pack()
        self.lbTC=tk.Label(self.root,text='当前时间:')
        self.lbTC.pack()

        # 包含当前日期和时间的datetime对象
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
        self.lbTC.config(text=dt_string)
  
        self.lb1=tk.Label(self.root,text='请选择您的爱好项目')
        self.lb1.pack()
        
        self.CheckVar1 =IntVar()
        self.CheckVar2 = IntVar()
        self.CheckVar3 = IntVar()
        self.CheckVar4 = IntVar()
        
        self.ch1 = tk.Checkbutton(self.root,text='古典',variable = self.CheckVar1,onvalue=11,offvalue=-11)
        self.ch2 = tk.Checkbutton(self.root,text='民谣',variable = self.CheckVar2,onvalue=12,offvalue=-12)
        self.ch3 = tk.Checkbutton(self.root,text='流行',variable = self.CheckVar3,onvalue=13,offvalue=-13)
        self.ch4 = tk.Checkbutton(self.root,text='爵士',variable = self.CheckVar4,onvalue=14,offvalue=-14)
        
        self.ch1.pack()
        self.ch2.pack()
        self.ch3.pack()
        self.ch4.pack()
        
        self.btn = tk.Button(self.root,text="搜索",command=self.run)
        self.btn.pack()
        
        self.lb2 = tk.Label(self.root,text='')
        self.lb2.pack()

        #btn = tk.Button(self.root, text="推荐音乐", command=self.classify)#推荐音乐按钮和点击事件
        #btn.pack(fill="both", expand=False, padx=10, pady=10)

        self.state = False
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)

        self.toggle_fullscreen(self)

        self.video_loop()
    
    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"
    def Mysel(self):
      dic = {0:'卧室',1:'客厅',2:'室内办公地点',3:'浴室'}
      s = "您选了" + dic.get(self.var.get()) + "项"
      self.lb2.config(text = s)
    def run(self):
        
        if(self.CheckVar1.get()==0 and self.CheckVar2.get()==0 and self.CheckVar3.get()==0 and self.CheckVar4.get()==0):
            s = '您还没选择任何爱好项目'
        else:
            s1 = "古典" if self.CheckVar1.get()==11 else ""
            s2 = "民谣" if self.CheckVar2.get() == 12 else ""
            s3 = "流行" if self.CheckVar3.get() == 13 else ""
            s4 = "爵士" if self.CheckVar4.get() == 14 else ""
            s = "您选择了%s %s %s %s" % (s1,s2,s3,s4)
        self.lb2.config(text=s)

    def video_loop(self):

        img, frame = self.vs.read()
        image = frame.copy()
        if img:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, 1)
            faces = self.faceCascade.detectMultiScale(img_gray,
                                                      scaleFactor=1.1,
                                                      minNeighbors=5,
                                                      minSize=(30, 30),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)

            emotions = []
            for (x, y, w, h) in faces:
                face_image_gray = img_gray[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                self.emt = self.predict_emotion(face_image_gray / 255)
                self.lbEC.config(text=self.emt)

                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, self.emt, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)

            self.current_image = Image.fromarray(frame).resize((300, 300), Image.ANTIALIAS)
            
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
        self.root.after(30, self.video_loop)

    def predict_emotion2(self, face_image_gray):  # a single cropped face
        resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(str(index)+'.png', resized_img)
        image = resized_img.reshape(1, 48, 48, 1)
        list_of_list = self.model.predict(image, batch_size=1)
        sad, neutral, happy, angry = [prob for lst in list_of_list for prob in lst]
        return [sad, neutral, happy, angry]
    def predict_emotion(self,face_image_gray):  # a single cropped face
        lab = ['angry', 'disgust', 'fear', 'happy','sad','surprise','neutral']
        resized_img = cv2.resize(face_image_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(str(index)+'.png', resized_img)
        image = resized_img.reshape(1, 48, 48, 1)
        image = image.astype(np.float32)
        #list_of_list = model.predict(image, batch_size=1)
        #print('1')
        interpreter = tf.lite.Interpreter(model_path="fer2013_mini_XCEPTION.119-0.65.tflite")
        interpreter.allocate_tensors()
        #print('2')
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # 模型预测
        #print('3')
        interpreter.set_tensor(input_details[0]['index'], image)  # 传入的数据必须为ndarray类型
        #print('4')
        interpreter.invoke()
        #print('5')
        output_data = interpreter.get_tensor(output_details[0]['index'])
    
        # 标签预测
        w = np.argmax(output_data)  # 值最大的位置
        lable_list = lab  # 读取标签列表

        #print(lable_list[w])
        return lable_list[w]

    def clear(self):
        self.rd1.destroy()
        self.rd2.destroy()
        self.rd3.destroy()
        self.rd4.destroy()
        self.rd5.destroy()
        self.button.destroy()

    def play(self):

        webbrowser.open("https://y.qq.com/n/ryqq/search?w={}".format(self.v.get().replace(" ", "+")))

    def classify(self):
        # self.current_image.save(".\{}".format("abc.jpg"))

        if self.emt == "sad":
            file = pd.read_csv("Warm.csv")
        elif self.emt == "neutral":
            file = pd.read_csv("Calm.csv")
        elif self.emt == "happy":
            file = pd.read_csv("Happy.csv")
        elif self.emt == "angry":
            file = pd.read_csv("Energetic.csv")
        music = file.sample(5)
        music.reset_index(inplace=True)

        self.v = tk.StringVar(self.root, "1")
        self.rd1 = tk.Radiobutton(self.root, text=music.music_name[0], variable=self.v, value=music.music_name[0],
                                  command=self.play)
        self.rd1.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd2 = tk.Radiobutton(self.root, text=music.music_name[1], variable=self.v, value=music.music_name[1],
                                  command=self.play)
        self.rd2.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd3 = tk.Radiobutton(self.root, text=music.music_name[2], variable=self.v, value=music.music_name[2],
                                  command=self.play)
        self.rd3.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd4 = tk.Radiobutton(self.root, text=music.music_name[3], variable=self.v, value=music.music_name[3],
                                  command=self.play)
        self.rd4.pack(fill="both", expand=True, padx=10, pady=1)
        self.rd5 = tk.Radiobutton(self.root, text=music.music_name[4], variable=self.v, value=music.music_name[4],
                                  command=self.play)
        self.rd5.pack(fill="both", expand=True, padx=10, pady=1)

        self.button = tk.Button(self.root, text="重置", command=self.clear)
        self.button.pack(fill="both", expand=True, padx=10, pady=1)

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


# start the app
print("[INFO] starting...")
pba = EmotionDetection()
pba.root.mainloop()

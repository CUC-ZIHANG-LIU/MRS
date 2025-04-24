import sys
import time
from PIL import Image, ImageTk
import datetime
import cv2
import pandas as pd
import numpy as np
import webbrowser
import warnings
import tensorflow as tf
import tkinter as tk
from tkinter import IntVar, filedialog

from datetime import datetime

warnings.filterwarnings("ignore")


class EmotionDetection:
  
    def __init__(self, ):
        self.emt=""
        self.search_time=[]
        self.state = False
        self.root = tk.Tk()
        self.panel = tk.Label(self.root)
        self.root.title("音乐推荐系统")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.panel.pack(padx=10, pady=10)
        # self.toggle_fullscreen(self)
        

       ####ME######
       
        #（列名 1-悲伤，2-安静，3-治愈，4-放松，5-愉快，6-令人激动， 行名 1-卧室，2-客厅，3-办公地点，4-浴室）
        self.lm = [0,0,0,0,0,0,0,
            0,0,0.25,0.15,0.4,0.2,0,
            0,0,0.14,0.13,0.28,0.32,0.13,
            0,0,0.28,0,0.4,0.32,0,
            0,0,0,0,0.6,0.4,0]
        #（列名 1-悲伤，2-安静，3-治愈，4-放松，5-愉快，6-令人激动， 行名 1-愤怒，2-厌恶，3-恐惧，4-开心，5-悲伤，6-惊讶，7-自然）
        self.pm = [0,0,0,0,0,0,0,
            0,0,0.33,0.33,0.34,0,0,
            0,0,0.3,0.3,0.4,0,0,
            0,0,0.3,0.25,0.45,0,0,
            0,0,0,0,0.29,0.44,0.27,
            0,0.2,0.2,0.35,0.25,0,0,
            0,0,0,0.33,0.28,0.39,0,
            0,0,0.2,0,0.35,0.25,0.2,]
        self.tm = [0,0,0,0,0,0,0,
            0,0.08,0.2,0.06,0.31,0.33,1,
            0,0.07,0.26,0.05,0.26,0.26,0.1,
            0,0.06,0.2,0.05,0.25,0.34,0.1,
            0,0.12,0.33,0.1,0.2,0.2,0.05]
        self.MM = [0,0.1,0.3,0.45,0.6,0.8,0.95]
        #C = [0,0.2,0.45,0.55,0.65,0.775,0.925]
        #A表示地点和歌曲情绪之间的对应矩阵4*7，不能用0检索（数根据调研结果改）
        self.LM = np.array(self.lm).reshape(5,7)
        #B表示人的情绪和歌曲情绪之间的对应矩阵7*7，不能用0检索（数根据调研结果改）
        self.PM = np.array(self.pm).reshape(8,7)
        self.TM = np.array(self.tm).reshape(5,7)
       ##########

        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy','sad','surprise','neutral']
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

        

 
        self.lbE=tk.Label(self.root,text='用户情绪:')
        self.lbE.pack()
        self.lbEC=tk.Label(self.root,text='[]')
        self.lbEC.pack()

        self.lbL=tk.Label(self.root,text='用户地点:')
        self.lbL.pack()
        self.lovar = IntVar()
        self.rd1 = tk.Radiobutton(self.root,text="卧室",variable=self.lovar,value=1,command=self.Mysel)
        self.rd1.pack()
        self.rd2 = tk.Radiobutton(self.root,text="客厅",variable=self.lovar,value=2,command=self.Mysel)
        self.rd2.pack()
        self.rd3 = tk.Radiobutton(self.root,text="室内办公地点",variable=self.lovar,value=3,command=self.Mysel)
        self.rd3.pack()
        self.rd4 = tk.Radiobutton(self.root,text="浴室",variable=self.lovar,value=4,command=self.Mysel)
        self.rd4.pack()

        self.lbT=tk.Label(self.root,text='当前时间:')
        self.lbT.pack()
        self.lbTC=tk.Label(self.root,text='当前时间:')
        self.lbTC.pack()


        self.lbTC.config(text=self.dt_string())
  
        self.lb1=tk.Label(self.root,text='请选择您的爱好项目')
        self.lb1.pack()
        
        self.CheckVar1 = IntVar()
        self.CheckVar2 = IntVar()
        self.CheckVar3 = IntVar()
        self.CheckVar4 = IntVar()
        self.CheckVar5 = IntVar()
        self.CheckVar6 = IntVar()
        self.CheckVar7 = IntVar()
        
        self.ch1 = tk.Checkbutton(self.root,text='古典',variable = self.CheckVar1,onvalue=11,offvalue=-11)
        self.ch2 = tk.Checkbutton(self.root,text='流行',variable = self.CheckVar2,onvalue=12,offvalue=-12)
        self.ch3 = tk.Checkbutton(self.root,text='摇滚',variable = self.CheckVar3,onvalue=13,offvalue=-13)
        self.ch4 = tk.Checkbutton(self.root,text='民谣',variable = self.CheckVar4,onvalue=14,offvalue=-14)
        self.ch5 = tk.Checkbutton(self.root,text='爵士',variable = self.CheckVar5,onvalue=15,offvalue=-15)
        self.ch6 = tk.Checkbutton(self.root,text='电子',variable = self.CheckVar6,onvalue=16,offvalue=-16)
        self.ch7 = tk.Checkbutton(self.root,text='轻音乐',variable = self.CheckVar7,onvalue=17,offvalue=-17)
        
        self.ch1.pack()
        self.ch2.pack()
        self.ch3.pack()
        self.ch4.pack()
        self.ch5.pack()
        self.ch6.pack()
        self.ch7.pack()
        
        self.btn = tk.Button(self.root,text="搜索",command=self.run)
        self.btn.pack()
        
        self.lb2 = tk.Label(self.root,text='')
        self.lb2.pack()

        self.lb3 = tk.Label(self.root,text='')
        self.lb3.pack()

        #btn = tk.Button(self.root, text="推荐音乐", command=self.classify)#推荐音乐按钮和点击事件
        #btn.pack(fill="both", expand=False, padx=10, pady=10)

        
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)

        

        self.video_loop()
    ############################################################################################################################
    def get_location(self):
        return str(self.lovar.get()) 

    def get_personemo(self):
        if(self.emt==""):
            return ""
        return str(self.emotion_labels.index(self.emt)+1)
    def get_timeorder(self):
        #时间分区间，共4个区间
        t = time.localtime()
        i = t.tm_hour
        if i>=5 and i<11:
            a = 1
        elif i>=11 and i<17:
            a = 2
        elif i>=17 and i<23:
            a = 3
        else :
            a = 4
        #print(a)
        return a

    def get_lo_score(self,location):
        #地点→音乐情绪
        c = []
        d = []
        for i in range(1, 7):
            if (self.LM[int(location)][int(i)] != 0):
                c.append(i)
                d.append(self.LM[int(location)][int(i)])
        # print(c)
        # print(d)
        #np.random.seed(i)
        p = np.array(d)
        index = np.random.choice(c, p=p.ravel())
        # print(index)
        resultlo = self.MM[index]
        return resultlo

    def get_personemo_score(self,personemo):
        #情绪→音乐情绪
        c = []
        d = []
        for i in range(1, 7):
            if (self.PM[int(personemo)][int(i)] != 0):
                c.append(i)
                d.append(self.PM[int(personemo)][int(i)])
        # print(c)
        # print(d)
        #np.random.seed(i)
        p = np.array(d)
        index = np.random.choice(c, p = p.ravel())
        # print(index)
        resultemo = self.MM[index]
        return resultemo

    def get_time_score(self,time):
        #时间→音乐情绪
        c = []
        d = []
        for i in range(1, 7):
            if (self.TM[int(time)][int(i)] != 0):
                c.append(i)
                d.append(self.TM[int(time)][int(i)])
        
        #np.random.seed(i)
        p = np.array(d)
        p /= p.sum()  # normalize https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
        index = np.random.choice(c, p=p.ravel())
        # print(index)
        resultlo = self.MM[index]
        return resultlo


    def record(self,lo_score,personemo_score,time_score):
        #计算最终要推荐的情绪评分
        weight = [0.2,0.7,0.1] #根据实验结果
        #score = lo_score * float(weight[0]) + personemo_score * float(weight[1]) + time_score * float(weight[2])
        score = lo_score * 0.2 + personemo_score * 0.7 + time_score * 0.1
        return score
    def Judge(self,score):
        #具体score评级与上面相同
        #非正常情况判断
        if score<=0:
            print("fail1")
            print(score)
            sys.exit(0)
        elif score>=1:
            print("fail1")
            sys.exit(0)
        #正常情况
        else :
            if score<0.2 :
                return "悲伤"
            elif score>=0.2 and score<0.4 :
                return "安静"
            elif score>=0.4 and score<0.5 :
                return "治愈"
            elif score>=0.5 and score<0.7 :
                return "放松"
            elif score>=0.7 and score<0.9 :
                return "愉快"
            else :
                return "令人激动的"
        
    def flavor(self,location,emotion,time): #传的参数待定
        #推荐音乐风格阶段，先判断用户是否选择合适的想要收听音乐风格（流行、摇滚等）数量，至少选1种，最多选3种。
        # 当选1种时，被推荐的风格就是所选择的。
        # 当选择2种时，推荐时先查询选择的风格在当前状态下总收听次数是否大于20，
        # 如果不大于，2种所选择风格概率都为50%，
        # 如果大于20，则通过 （该类所听次数÷总听次数） 获得2种风格被推荐概率。
        # 当选择3种时，判断表内是否大于30，不大于的话每种都是1/3，大于的话处理方式与选2种相同。

        if(len(self.like_list))==1:
            return self.like_list[0]
        
        with open("user.csv") as file_name:#打开列表，注意这里的encoding参数应为CBK
            file_name.readline()#对首行的忽略
            file_obj=file_name.readlines()#将下面的内容（每行）转化为列表的形式返回
            for line in file_obj:#遍历列表中的元素
                lists=line.rstrip("\n").split(",")#首先用strip去掉换行符，再用sprit分割从而的到多个单词
                if self.etl==lists[0] :
                    break;#找到了

        style_list=["古典","流行","摇滚","民谣","爵士","电子","轻音乐"]
        

        if(len(self.like_list))==2:
            #查询选择的风格在当前状态下总收听次数是否大于20
            str_num1=lists[style_list.index(self.like_list[0])+1]
            str_num2=lists[style_list.index(self.like_list[1])+1]
            if(str_num1==""):str_num1="0"
            if(str_num2==""):str_num2="0"
            num1=int(float(str_num1))
            num2=int(float(str_num2))
            totle=num1+num2
            if(totle>20):
                p = np.array([num1/totle,num2/totle])
            else:
                p = np.array([0.5,0.5])
            resultstyle = np.random.choice(self.like_list, p=p.ravel())
        if(len(self.like_list))==3:
            #查询选择的风格在当前状态下总收听次数是否大于30
            str_num1=lists[style_list.index(self.like_list[0])+1]
            str_num2=lists[style_list.index(self.like_list[1])+1]
            str_num3=lists[style_list.index(self.like_list[2])+1]
            if(str_num1==""):str_num1="0"
            if(str_num2==""):str_num2="0"
            if(str_num3==""):str_num3="0"
            num1=int(float(str_num1))
            num2=int(float(str_num2))
            num3=int(float(str_num3))
            totle=num1+num2+num3
            if(totle>30):
                p = np.array([num1/totle,num2/totle,num3/totle])
            else:
                p = np.array([1/3,1/3,1/3])
            resultstyle = np.random.choice(self.like_list, p=p.ravel())
 
        #如果查到了返回查到的音乐
        #MS查表推荐

        return resultstyle
    ############################################################################################################################

    def dt_string(self):
        # 包含当前日期和时间的datetime对象
        now = datetime.now()
        # dd/mm/YY H:M:S
        return now.strftime("%Y-%m-%d %H:%M:%S")
    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"
    def Mysel(self):
      dic = {1:'卧室',2:'客厅',3:'室内办公地点',4:'浴室'}
      s = "您选了" + dic.get(self.lovar.get()) + "项"
      self.lb2.config(text = s)
    def get_time(self,a1, a2):
        timeArraya1 = time.strptime(a1, "%Y-%m-%d %H:%M:%S")
        # 转换为时间戳
        timeStampa1 = int(time.mktime(timeArraya1))

        timeArraya2 = time.strptime(a2, "%Y-%m-%d %H:%M:%S")
        # 转换为时间戳
        timeStampa2 = int(time.mktime(timeArraya2))

        x = timeStampa2 - timeStampa1
        return x
    def add_num(self,etl,style):
        # 读取文件
        data = pd.read_csv("user.csv")
        df = pd.DataFrame(data)
        df.set_index('E/T/L',inplace=True)
        str_num=df.at[int(etl),str(style)]
        if(str_num=="" or str_num != str_num):
            num=0
        else:
            num=int(str_num)
        df.at[int(etl),str(style)] = str(num+1)
        #生成新的文件
        data.to_csv("user.csv")
    #搜索处理事件
    def run(self):
        self.search_time.append(self.dt_string())#保存搜索时间
        self.lbTC.config(text=self.dt_string())
        self.like_list = []  
        if(self.CheckVar1.get()==0 and self.CheckVar2.get()==0 and self.CheckVar3.get()==0 and self.CheckVar4.get()==0 and self.CheckVar5.get()==0 and self.CheckVar6.get()==0 and self.CheckVar7.get()==0):
            s = '您还没选择任何爱好项目'
            self.lb2.config(text=s)
            return
        else:
            s1 = "古典" if self.CheckVar1.get()==11 else ""
            s2 = "流行" if self.CheckVar2.get() == 12 else ""
            s3 = "摇滚" if self.CheckVar3.get() == 13 else ""
            s4 = "民谣" if self.CheckVar4.get() == 14 else ""
            s5 = "爵士" if self.CheckVar5.get() == 15 else ""
            s6 = "电子" if self.CheckVar6.get() == 16 else ""
            s7 = "轻音乐" if self.CheckVar7.get() == 17 else ""
            #最多选三个
            if s1 !="" :self.like_list.append(s1)
            if s2 !="" :self.like_list.append(s2)
            if s3 !="" :self.like_list.append(s3)
            if s4 !="" :self.like_list.append(s4)
            if s5 !="" :self.like_list.append(s5)
            if s6 !="" :self.like_list.append(s6)
            if s7 !="" :self.like_list.append(s7)

            if(len(self.like_list)>3):
                s="最多选三个"
                self.lb2.config(text=s)
                return
            else:
                s = "您选择了%s %s %s %s %s %s %s" % (s1,s2,s3,s4,s5,s6,s7)
        self.lb2.config(text=s)
        #推荐逻辑
        #rec=""
        
        
        person_emo = self.get_personemo()
        time1 = self.get_timeorder()
        lo = self.get_location()

        if(person_emo==""):
            self.lb2.config(text="获取情绪失败，请检查摄像头")
            return
        if(lo=="0"):
            self.lb2.config(text="请选择地点")
            return

        #异常检查完毕

        #引入一个判断每次推荐是否有效模块，在推荐时计时，若下一次再点击推荐间隔大于30秒，则判断成功，
        # 根据推荐时所包含的状态编码，在用户表中找到对应的行列，在对应位置做+1操作。如果推荐间隔小于30，则不记录。
        search_num=len(self.search_time)
        if(search_num>1):
            x=self.get_time(self.search_time[search_num-2],self.search_time[search_num-1])
            if(x>30):
                print(">30")
                self.add_num(self.last_etl,self.last_style)
            else:
                print("<30")
                #self.add_num(self.last_etl,self.last_style)#debug tobe del



        self.etl=person_emo+str(time1)+lo
        print("ETL=",self.etl)

        lo_score = self.get_lo_score(lo)
        # print('lo_score:',lo_score)
        personemo_score = self.get_personemo_score(person_emo)
        # print('personemo_score:',personemo_score)
        time_score = self.get_time_score(time1)
        # print('time_score:',time_score)
        score1 = self.record(lo_score,personemo_score,time_score)
        # print(score1)
        music_emo = self.Judge(score1)#音乐情绪
        print(music_emo)

        style1 = self.flavor(lo,person_emo,time1)#ms 音乐风格
        self.ms=style1#保存音乐风格
        print(self.ms)

        # print(style1)
        rec="推荐音乐："+music_emo+" "+style1 #联合搜索

        self.last_etl=self.etl
        self.last_style=style1

        self.lb3.config(text=rec)

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

    
    

    def destructor(self):
        """ Destroy the root object and release all resources """
        # print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


# start the app
# print("[INFO] starting...")
pba = EmotionDetection()
pba.root.mainloop()

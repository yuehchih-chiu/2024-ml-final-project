from tensorflow.keras.models import model_from_json
from PIL import Image,ImageDraw,ImageGrab, ImageTk
from sympy import *
from tkinter import*
import cv2
import numpy as np

#########
from tkinter import Variable, filedialog
########

#載入模型
print('Loading Model...')
model_json = open('model/model.json', 'r')
loaded_model_json = model_json.read()
model_json.close()
model = model_from_json(loaded_model_json)

#載入權重
print('Loading weights...')
model.load_weights("model/model_weights.h5")

labels = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','+','-','x']

"""
這段程式碼用於從磁碟中載入訓練好的模型。讓我們逐行解釋：

print('Loading Model...')：這行程式碼只是打印一個訊息，用來提示正在載入模型。
model_json = open('model/model.json', 'r')：
這行程式碼打開了一個檔案，該檔案包含了模型的結構（通常以 JSON 格式保存）。檔案名為 'model/model.json'，表示它位於名為 model 的資料夾中。
loaded_model_json = model_json.read()：這行程式碼讀取了打開的檔案中的內容，並將其存儲在變數 loaded_model_json 中。
model_json.close()：這行程式碼關閉了打開的檔案，以釋放系統資源。
model = model_from_json(loaded_model_json)：
這行程式碼使用 Keras 的 model_from_json 函式將 JSON 格式的模型結構轉換為 Keras 模型物件。
loaded_model_json 變數包含了模型的結構資訊，通過這個函式，我們將這些資訊轉換成了一個可用的 Keras 模型。
"""




##############################################################

def activate_event(event):
        global lasx,lasy
        lasx,lasy=event.x,event.y

"""
聲明了 lasx 和 lasy 這兩個變數是全局變數
event.x 和 event.y 屬性獲取滑鼠當前的 x 和 y 座標
"""


def draw_smth(event):
    global lasx,lasy
    cv.create_line((lasx,lasy,event.x,event.y),fill='black',width=4)
    lasx,lasy=event.x,event.y

"""
這個函式的操作是：

使用 cv.create_line() 方法在畫布上創建一條線段，起點是 (lasx, lasy)，終點是 (event.x, event.y)。
更新全局變數 lasx 和 lasy 為當前滑鼠的位置，以便在下一次函式調用時使用。
這樣，當用戶在畫布上移動滑鼠時，就會在畫布上畫出連續的線條，從上一個滑鼠位置到當前的滑鼠位置。
"""

def save():
    filename="canvas.jpg"
    widget=cv
    x = root.winfo_rootx() + widget.winfo_x()+50
    y = root.winfo_rooty() + widget.winfo_y()+50
    x1=x+1600
    y1=y+700
    """
    
    x1=x+widget.winfo_width()
    y1=y+widget.winfo_height()
    """
    ImageGrab.grab().crop((x,y,x1,y1)).save(filename)

"""
 widget 這個變數代表畫布 cv
 
 x=root.winfo_rootx()+widget.winfo_x()+50：
 這行程式碼計算了畫布 cv 左上角的 x 座標。
 root.winfo_rootx() 返回了窗口的左上角相對於屏幕左上角的 x 座標，
 widget.winfo_x() 返回了畫布 cv 在窗口中的 x 座標，然後再加上偏移量 50。
 
 x1=x+widget.winfo_width()：
 這行程式碼計算了畫布 cv 右下角的 x 座標
 
 使用 ImageGrab.grab().crop() 函式截取屏幕上指定區域的圖片，並將其保存為檔案
 
"""


def predictFromArray(arr):
    result = np.argmax(model.predict(arr), axis=-1)
    return result


def identify_super_subscript(en,rn):

    #print(len(en))
    #print(len(rn))

    en1=""

    #for i in range(len(en)):
    i=0
    while i<=(len(en)-1):

        if(en[i]=='-' or en[i]=='='):
            en1+=en[i]
            i += 1
            continue

        subscript = ""
        superscript = ""

        a_x, a_y, a_w, a_h = rn[i][0], rn[i][1], rn[i][2], rn[i][3]

        flagsub = 0
        flagsuper = 0

        #print('i=')
        #print(i)

        i_temp =i

        #for j in range(i+1,len(en)):
        j = i+1
        while j <= (len(en) - 1):

            #print('j=')
            #print(j)

            x, y, w, h = rn[j][0], rn[j][1], rn[j][2], rn[j][3]
            tempy = y + 0.5 * h

            if (tempy > a_y + 0.75 * a_h):
                #print('Hi1')
                subscript+=en[j]
                flagsub = 1
                i+= 1

            elif (tempy < a_y + 0.25 * a_h):
                #print('Hi2')
                superscript+=en[j]
                flagsuper = 1
                i += 1

            else:
                break

            j += 1

        en1+=en[i_temp]

        if(flagsub == 1):
            #print('Hi3')
            en1+="_{"
            en1+=subscript
            en1+="}"

        if (flagsuper == 1):
            #print('Hi4')
            en1+="^{"
            en1+=superscript
            en1+="}"

        i += 1


    return en1


####################################################




def identify_fraction(en,rn):

    final_en=""

    flag1=0

    #for i in range(len(en)):
    i=0
    while i<=(len(en)-1):


        print('i=')
        print(i)

        numerator = []
        numerator_rect = []
        denominator = []
        denominator_rect = []

        flag = 0

        nf = en
        nf_r = rn
        #en1=""


        if(en[i]=='-'):


            a_x, a_y, a_w, a_h = rn[i][0], rn[i][1], rn[i][2], rn[i][3]
            a_x_1=a_x - 0.1 * a_w
            a_x_2 = a_x + a_w + 0.1 * a_w

            # for j in range(i+1,len(en)):
            #j = i + 1
            j=0

            #len(en)  底下j迴圈
            while j <= (len(en) - 1):

                print('j=')
                print(j)

                x, y, w, h = rn[j][0], rn[j][1], rn[j][2], rn[j][3]
                tempx = x + 0.5 * w
                tempy = y + 0.5 * h


                if( x==a_x and y==a_y ):
                    j += 1
                    continue

                if ( a_x_1 <  tempx and tempx  < a_x_2):

                    flag = 1
                    nf[i] = "n"
                    #nf_r[i] = "n"

                    if (tempy < a_y):

                        numerator.append(en[j])
                        numerator_rect.append(rn[j])
                        i += 1

                    elif (tempy > a_y):

                        denominator.append(en[j])
                        denominator_rect.append(rn[j])
                        i += 1

                    nf[j]="n"
                    #nf_r[j]="n"

                j += 1


        nf1 = []
        nf_r1 = []

        if(flag1 == 0):
            iitemp=i

        print("iitemp=")
        print(iitemp)

        for ii in range(iitemp,i+1):
            print('ii=')
            print(ii)
            if (nf[ii] == "n"):
                break
            nf1.append(nf[ii])
            nf_r1.append(nf_r[ii])

        print(nf)
        print(nf_r)
        print(nf1)
        print(nf_r1)
        print(numerator)
        print(numerator_rect)
        print(denominator)
        print(denominator_rect)

        flag1 = 1

        if (flag == 1):
            en1 = ""
            en1 += identify_super_subscript(nf1, nf_r1)
            en1 += "\\"
            en1 += "frac{"
            en1 += identify_super_subscript(numerator, numerator_rect)
            en1 += "}{"
            en1 += identify_super_subscript(denominator, denominator_rect)
            en1 += "}"
            final_en += en1
            flag1=0


        i += 1


    if (flag == 0):
        en1 = ""
        en1 += identify_super_subscript(nf1, nf_r1)
        final_en += en1

    return final_en

##################################################################


##識別照片

def solution():
    ###'canvas.jpg'即是我們手寫出來的照片，會一直更新

    img = cv2.imread('canvas.jpg', cv2.IMREAD_GRAYSCALE)
    # 反轉圖像
    img = ~img
    # 圖像二值化
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 尋找輪廓
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排序輪廓
    cnt = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # 對每個輪廓進行處理

    img_data = []
    rects = []
    for c in cnt:
        x, y, w, h = cv2.boundingRect(c)# 每個輪廓的位置
        rect = [x, y, w, h]
        rects.append(rect)
    final_rect = [i for i in rects]


    # 對每個輪廓進行預處理並添加到 img_data 中

    for r in final_rect:
        x, y, w, h = r[0], r[1], r[2], r[3]
        img = thresh[y:y + h + 10, x:x + w + 10]
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img, (1, 28, 28))
        img_data.append(img)

    # 識別手寫數字和符號  使用predictFromArray進行預測

    mainEquation = []
    operation = ''
    for i in range(len(img_data)):
        img_data[i] = np.array(img_data[i])
        img_data[i] = img_data[i].reshape(-1, 28, 28, 1)
        result = predictFromArray(img_data[i])
        i = result[0]
        mainEquation.append(labels[i])



    ######################################################
    img2 = cv2.imread('canvas.jpg')

    j=0
    for r in final_rect:
        x, y, w, h = r[0], r[1], r[2], r[3]
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img2, mainEquation[j], (x - 20, y), cv2.FONT_HERSHEY_TRIPLEX,1, (0, 0, 255))

        cv2.circle(img2, (int(x + 0.5*w), int(y + 0.5*h)), 7, (0,255,0), -1)
        cv2.putText(img2, str(x + 0.5*w)+" , "+str(y + 0.5*h) , (int(x + 0.5*w), int(y + 0.5*h+20)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255,0))


        j=j+1
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow('img2', img2)
    #############################################################




    print(mainEquation)
    print(final_rect)

    #######def identify_equ():
    for i in range(len(mainEquation)-1):
        a = mainEquation[i]

        if (a=='-'):

            a_x, a_y, a_w, a_h = final_rect[i][0],final_rect[i][1],final_rect[i][2],final_rect[i][3]
            #print(a_x, a_y, a_w, a_h)

            if(mainEquation[i+1]=='-'):
                if(abs(a_w - final_rect[i+1][2]) < 0.5*a_w and abs(a_w - final_rect[i+1][2]) < 0.5*final_rect[i+1][2]):
                    #print('Hi')
                    if(abs(a_x - final_rect[i+1][0]) < 0.5*a_w and abs(a_y - final_rect[i+1][1]) < 0.5*a_w):
                        #print('Hi')
                        mainEquation[i]='='
                        del mainEquation[i+1]
                        del final_rect[i+1]

    print(mainEquation)
    print(final_rect)



    mainEquation1=identify_fraction(mainEquation,final_rect)
    print(mainEquation1)







    cve2.configure(text='Your Equation is : ' +mainEquation1)#打印結果



"""打印是equ  純數字"""
####################################################





################################################


def show():
    import tkinter as tk
    from tkinter import Variable, filedialog
    from PIL import Image, ImageTk
    img_path = filedialog.askopenfilename(filetypes=[('png', '*.png'),('jpg', '*.jpg'),('gif', '*.gif')])  # 指定開啟檔案格式
    img22 = Image.open(img_path)           # 取得圖片路徑
    w, h = img22.size
    # 取得圖片長寬
    #resized_image = image.resize((new_width, new_height))
    new_img22 = img22.resize((int(w*2/3), int(h*2/3)))
    tk_img = ImageTk.PhotoImage(new_img22)     # 轉換成 tk 圖片物件
    cv.delete('all')                 # 清空 Canvas 原本內容
    #cv.config(scrollregion=(0,0,w/3,h/3))   # 改變捲動區域
    cv.create_image(0, 0, anchor='nw', image=tk_img)   # 建立圖片
    cv.tk_img = tk_img               # 修改屬性更新畫面



#################################################




#設置 tkinter 視窗
root=Tk()
root.resizable(0,0)
root.title('Equation Solver')

lasx,lasy=None,None

cv=Canvas(root,width=1200,height=500,bg='white')
#cv=Canvas(root,width=1600,height=500,bg='white')
cv.grid(row=0,column=0,pady=2,sticky=W,columnspan=2)
cve2=Label(root,font=("Helvetica",16))
#cve=Label(root,font=("Helvetica",16))
#cve.grid(row=0, column=1,pady=1, padx=1)
cve2.grid(row=1, column=1,pady=1, padx=1)

cv.bind('<Button-1>',activate_event)
cv.bind('<B1-Motion>',draw_smth)


"""
*畫線
"""
btn_save=Button(text="Save",command=save,bg='#6495ED',fg='White')
btn_save.grid(row=2,column=0,pady=1,padx=1)
"""
*save
存照片
"""

btn_predict=Button(text="Predict",command=solution,bg='#6495ED',fg='White')
btn_predict.grid(row=2,column=1,pady=1,padx=1)

##########################開啟圖片
button_photo= Button(root, text='開啟圖片', command=show)
button_photo.grid(row=3,column=1,pady=1,padx=1)
#############################

"""
*solution
預測結果
"""


root.mainloop()

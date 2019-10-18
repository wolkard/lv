#-*- coding:utf-8-*-
from app import app
from flask import render_template,flash,redirect,request,jsonify
from werkzeug.utils import secure_filename
import os,base64
import cv2
import math
import time
#import mask_rcnn.lv_recognition as lv_recognition
import mask_rcnn.lv_recognition_skeleton as lv_recognition_skeleton
import json
import random
import numpy as np

@app.route('/lv_sk_info',methods=['GET'])
def lv():
    return render_template('sk_info.html') 

# 用于判断文件后缀
def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['png','jpg','JPG','PNG','jpeg','JPEG'])
    return '.' in filename and filename.rsplit('.',1)[1] in ALLOWED_EXTENSIONS
 

# 上传文件------------------------------------------------------------------------zan shi wei yong  dao
@app.route('/push',methods=['POST'],strict_slashes=False)
def api_upload():
    f=request.files['file']  # 从表单的file字段获取文件，myfile为该表单的name值
    if f and allowed_file(f.filename):  # 判断是否是允许上传的文件类型
        fname=secure_filename(f.filename)
        #ext = fname.rsplit('.',1)[1]  # 获取文件后缀
        unix_time = int(time.time())
        new_filename=str(unix_time)+"_"+str(random.randint(1,9999))+'.jpg'  # 修改了上传的文件名
        f_url = "../lv_web/app/static/imgs/o_img/"+new_filename        
        f.save(f_url)  #保存文件到upload目录
        return jsonify({"result":1,"remind":"上传成功","url":"static/imgs/o_img/"+new_filename,"img_name":new_filename})
    else:
        return jsonify({"result":0,"remind":"上传失败"})

@app.route('/lv_sk',methods=['GET','POST'])
def lv_sk():
    return render_template('sk_home.html') 

@app.route('/sk_push',methods=['POST'])
def sk_push():
    image = request.form.get('image')# 获取的编码后的img
    points = json.loads(request.form.get('points'))# 获取的编码后的坐标信息	
    image = image.split(";")[1].split(",")[1]
    try:
        img = base64.b64decode(image)
    except:
        result = jsonify({"res":-1})
    unix_time = int(time.time())
    new_filename=str(unix_time)+"_"+str(random.randint(1,9999))+'.jpg'# 创建文件名
    file_url = "../lv_web/app/static/imgs/o_img/"+new_filename
    file=open(file_url,'wb')# 保存文件到upload目录 
    file.write(img)
    file.close()

    img = cv2.imread(file_url)
    dst_pts = np.array([[0,0],[1920,0],[1920,1080],[0,1080]])
    src_pts = np.array(points)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    im_out = cv2.warpPerspective(img, np.linalg.inv(M), (1920, 1080))
    cv2.imwrite("../lv_web/app/static/imgs/img/"+new_filename,im_out)

    img_name = new_filename.split(".")[0]
    O_height,new_p_proportion_1_11,p_O_width,p_N_width,new_p_sum,best_like_img,best_result,num_out,num_surplus,res,remind = lv_recognition_skeleton.calculate(img_name)

    str_new_p_proportion_1_11='1'
    if res == 1 :#recognition right
        for i in new_p_proportion_1_11[1:]:
            str_new_p_proportion_1_11+=" : "+str(round(i,6))
    if(best_like_img!=0):
        aline(best_like_img,img_name)
    result = {"res":res,
            "remind":remind,
            "O_height":O_height,
            "new_p_proportion_1_11":str_new_p_proportion_1_11,
            "p_O_width":p_O_width,
            "p_N_width":p_N_width,
            "new_p_sum":new_p_sum,
            "logo_url":"static/imgs/logo/"+img_name+"_logo.jpg",
            "sk_img":"static/imgs/sk_img/"+img_name+"_sk.jpg",
            "best_result":best_result,
            "like":1/(1+math.exp(-(best_result-0.85)*30))*100,
            "best_like_img":"static/imgs/t_imgs/"+str(best_like_img),
            "best_like_sk_img":"static/imgs/t_sk_imgs/"+str(best_like_img),
        	"best_like_bw_img":"static/imgs/t_bw_imgs/"+str(best_like_img),
            "aline_sk_img":"static/imgs/aline_sk_imgs/"+str(img_name)+"_aline.jpg"
	        }
    return jsonify(result)

def aline(best_like_img,img_name):
    t_sk = cv2.imread("app/static/imgs/t_sk_imgs/"+best_like_img,0)
    sk = cv2.imread("app/static/imgs/sk_img/"+img_name+"_sk.jpg",0)
    t_sk_to_x = t_sk.sum(axis=0)
    t_sk_white_index = np.where(t_sk_to_x>10*255)[0]
    sk_to_x = sk.sum(axis=0)
    sk_white_index = np.where(sk_to_x>10*255)[0]

    t_sk_logo = t_sk[:,t_sk_white_index[0]-1:t_sk_white_index[-1]+2]
    t_sk_logo = cv2.resize(t_sk_logo, (980,115), interpolation=cv2.INTER_AREA)

    sk_logo = sk[:,sk_white_index[0]-1:sk_white_index[-1]+2]
    sk_logo = cv2.resize(sk_logo, (980,115), interpolation=cv2.INTER_AREA)
    
    t_sk_logo_copy = t_sk_logo.copy()
    t_sk_logo_copy[t_sk_logo_copy<100]=0
    t_sk_logo_copy[t_sk_logo_copy>=100]=1
    sk_logo_copy = sk_logo.copy()
    sk_logo_copy[sk_logo_copy<100]=0
    sk_logo_copy[sk_logo_copy>=100]=1
    like_all_line_segments,like_num_out,like_num_surplus,all_width = lv_recognition_skeleton.skeleton_calculation(sk_logo_copy,100,best_like_img)
    like_all_line_segments,like_num_out,like_num_surplus,like_all_width = lv_recognition_skeleton.skeleton_calculation(t_sk_logo_copy,100,best_like_img+"1")
    
    t_sk_logo=t_sk_logo[:,:,np.newaxis]
    t_sk_logo = t_sk_logo.repeat(3,axis=2)
    sk_logo = sk_logo[:,:,np.newaxis]
    sk_logo = sk_logo.repeat(3,axis=2)
    sk_logo[:,:,[2]] = 0
    sk_logo[:,:,[0,1]][sk_logo[:,:,[0,1]] > 150] = 255
    t_sk_logo[:,:,[1]]=0
    t_sk_logo[:,:,[0,2]][sk_logo[:,:,[0,2]] > 150] = 255
    #求出最近似图片的骨架图的所有点
    if t_sk_logo.shape[0]>sk_logo.shape[0] :
        new_sk_img = np.zeros((t_sk_logo.shape[0],1000,3))
    else:
        new_sk_img = np.zeros((sk_logo.shape[0],1000,3))
    new_sk_img[:t_sk_logo.shape[0],5:t_sk_logo.shape[1]+5] += t_sk_logo
    new_sk_img[:sk_logo.shape[0],5:sk_logo.shape[1]+5] += sk_logo
    new_sk_img[new_sk_img>255]=255
    new_sk_img = 255-new_sk_img
    cv2.imwrite("app/static/imgs/aline_sk_imgs/"+str(img_name)+"_aline.jpg",new_sk_img)
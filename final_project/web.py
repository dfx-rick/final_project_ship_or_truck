import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from model.helper.predict_helper import predict
from PIL import Image


app=Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
model=  tf.keras.models.load_model('saved_model/f_model.h5')

path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            basepath=os.path.dirname(__file__)
            file_path= os.path.join(basepath, 'uploads', filename)
            file.save(file_path)
            img=Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img= img.resize((32,32))
            res=predict(img, model)
            if(res== 0):
                return render_template('upload.html', prediction_text="The image given is a ship")
            elif(res==1):    
                return render_template('upload.html', prediction_text="The image given is a truck")

            
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


if __name__ == "__main__":
    app.run(debug = True)
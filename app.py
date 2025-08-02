from flask import Flask, render_template, request, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from model_predictor import check_sambar_consistency

app = Flask(__name__)
app.secret_key = 'sambar-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None

    if request.method == 'POST':
        img = request.files.get('sambar_image')
        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            result, _, _ = check_sambar_consistency(filepath)
        else:
            flash("Invalid file type. Please upload a valid image.")

    return render_template('index.html', result=result, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5000)

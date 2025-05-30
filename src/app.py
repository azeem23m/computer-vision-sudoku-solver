from flask import Flask, render_template, request, jsonify, send_file
from model import CNN
import os

app = Flask(__name__)
app.config['ALLOWED_EXTENTIONS'] = ['png', 'jpg', 'jpeg']
app.model = CNN()


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():

  file = request.files['puzzle']
  file_extension = file.filename.split('.')[-1]
  
  if file_extension not in app.config['ALLOWED_EXTENTIONS']:
    return jsonify({'error': 'Invalid file type'})
  
  file_path = os.path.join('images', file.filename)

  file.save(file_path)


  app.model.load_model('models/model.pth')  
  solved = app.model.solve_sudoku(file_path)
  if not solved:
    return render_template('fail.html')
  

  return send_file('images/solution.png', mimetype='image/png')

if __name__ == '__main__':
  app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response, jsonify, Response
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
import MySQLdb.cursors
import numpy as np
import numpy.random as rng
import pandas as pd
import requests
import re
import os
import io
from openpyxl import Workbook
from base64 import b64encode, b64decode
from PIL import Image
from keras import backend as K
import tensorflow as tf
from getpass import getpass

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY')
app.config['MYSQL_HOST'] = os.environ.get('DATABASE_HOST')
app.config['MYSQL_USER'] = os.environ.get('DATABASE_USER')
app.config['MYSQL_PASSWORD'] = os.environ.get('DATABASE_PASSWORD')
app.config['MYSQL_PORT'] = int(os.environ.get('DATABASE_PORT'))
app.config['MYSQL_DB'] = os.environ.get('DATABASE_NAME')

my_url = os.environ.get('HOST_URL')
app_dir = os.environ.get('APP_DIR')

mysql = MySQL(app)
UPLOAD_FOLDER = os.path.join(app_dir,'static','uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

class Image_Preprocessing:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width
    
    def processing(self, image):
        # Resize image -> Grayscale conversion -> Normalization
        image = image.resize((self.img_width, self.img_height))
        image = image.convert('L')
        image = np.array(image, dtype='float32')
        image = image / 255.0
        return image[..., np.newaxis]
    
    def imread(self, path):
        image = Image.open(path)
        image = self.processing(image)
        return image

img_pre = Image_Preprocessing(155, 220)

def get_batch(batch_size, room_id):
    response = requests.get(f"{my_url}/api/signatures/room/{room_id}")
    myresult = response.json()
    df = pd.DataFrame(myresult)
    signers_id = df['id'].unique()

    #randomly sample several classes to use in the batch
    n_signers = len(signers_id)
    categories = rng.choice(n_signers, size=(batch_size,), replace=False)

    #initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, img_pre.img_height, img_pre.img_width, 1)) for i in range(2)]

    #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
    targets=np.zeros((batch_size,))
    targets[batch_size//2:] = 1

    for i in range(batch_size):
        category_1 = categories[i]

        #pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category_1
        else: 
            #add a random number to the category modulo n classes to ensure 2nd image has different category
            category_2 = (category_1 + rng.randint(1, n_signers)) % n_signers

        images_1 = df[df['id'] == signers_id[category_1]]['signature_image']
        images_2 = df[df['id'] == signers_id[category_2]]['signature_image']

        rand_image_1 = img_pre.imread(io.BytesIO(b64decode(rng.choice(images_1))))
        rand_image_2 = img_pre.imread(io.BytesIO(b64decode(rng.choice(images_2))))

        pairs[0][i,:,:,:] = rand_image_1
        pairs[1][i,:,:,:] = rand_image_2
    return pairs, targets

def get_model(room_id):
    response = requests.get(f"{my_url}/api/models/{room_id}")
    model_data = response.json()
    model_path = os.path.join(app_dir,'static','models', model_data['model_name']+".h5")
    custom_objects = {"contrastive_loss": contrastive_loss, 'K':K}
    model = tf.keras.models.load_model(os.path.join(app_dir,'default_model.h5'), custom_objects)
    model.load_weights(model_path)
    return model


# accounts
@app.route('/api/accounts/<id>', methods=['GET'])
def take_account_by_id(id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts WHERE id = %s', (id,))
    account = cursor.fetchone()
    return make_response(jsonify(account), 200)

@app.route('/api/accounts/username/<username>', methods=['GET'])
def take_account_by_username(username):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
    account = cursor.fetchone()
    return make_response(jsonify(account), 200)

@app.route('/api/accounts/std_id/<string:std_id>', methods=['GET'])
def take_account_by_std_id(std_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts WHERE std_id = %s', (std_id,))
    account = cursor.fetchone()
    return make_response(jsonify(account), 200)

@app.route('/api/accounts/login/', methods=['GET'])
def take_account_by_login():
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (data['username'], data['password']))
    account = cursor.fetchone()
    return make_response(jsonify(account), 200)

@app.route('/api/accounts/', methods=['POST'])
def add_account():
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('INSERT INTO accounts (username, password, email, std_id, fname, lname) VALUES (%s, %s, %s, %s, %s, %s)', 
                   (data['username'], data['password'], data['email'], data['std_id'], data['fname'], data['lname']))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'You have successfully registered!'}), 200)

@app.route('/api/accounts/<id>', methods=['PUT'])
def change_account(id):
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('UPDATE accounts SET std_id = %s, fname = %s, lname = %s WHERE id = %s', 
                   (data['std_id'], data['fname'], data['lname'], id))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'You have successfully updated!'}), 200)


# signatues
@app.route('/api/signatures/<account_id>', methods=['GET'])
def take_signatures(account_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT signature_id, signature_image FROM signatures WHERE account_id = %s', (account_id,))
    signatures = cursor.fetchall()
    data = []
    for row in signatures:
        id = row['signature_id']
        image = row['signature_image']
        image = b64encode(image).decode('utf-8')
        data.append({'signature_id' : id, 'signature_image' : image})
    return make_response(jsonify(data), 200)

@app.route('/api/signatures/room/<room_id>', methods=['GET'])
def take_signatures_by_room(room_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT signature_image, id
    FROM signatures AS sig, accounts AS a, join_rooms AS jr, rooms AS r
    WHERE sig.account_id = a.id AND a.id = jr.account_id AND jr.room_id = r.room_id AND r.room_id = %s""", (room_id,))
    signatures = cursor.fetchall()
    data = []
    for row in signatures:
        id = row['id']
        image = row['signature_image']
        image = b64encode(image).decode('utf-8')
        data.append({'signature_image' : image, 'id' : id})
    return make_response(jsonify(data), 200)

@app.route('/api/signatures/std_id/<string:std_id>', methods=['GET'])
def take_signatures_by_std_id(std_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT signature_image FROM signatures, accounts WHERE std_id = %s AND account_id = id""", (std_id,))
    signatures = cursor.fetchall()
    data = []
    for row in signatures:
        image = row['signature_image']
        image = b64encode(image).decode('utf-8')
        data.append({'signature_image' : image})
    return make_response(jsonify(data), 200)

@app.route('/api/signatures/', methods=['POST'])
def add_signature():
    data = request.get_json()
    image = b64decode(bytes(data['signature_image'], 'utf-8'))
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('INSERT INTO signatures (signature_image, account_id) VALUES (%s, %s)', 
                   (image, data['account_id']))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Upload image successfully!'}), 201)

@app.route('/api/signatures/<signature_id>', methods=['DELETE'])
def erase_signature(signature_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('DELETE FROM signatures WHERE signature_id = %s', (signature_id,))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Delete image successfully!'}), 200)


# room
@app.route('/api/rooms/', methods=['GET'])
def take_rooms():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT room_id, room_name, description FROM rooms')
    rooms = cursor.fetchall()
    return make_response(jsonify(rooms), 200)

@app.route('/api/rooms/<room_id>', methods=['GET'])
def take_room_by_id(room_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""SELECT r.room_id, room_name, description, train_status 
    FROM rooms AS r, models AS m 
    WHERE r.room_id = m.room_id AND r.room_id = %s""", (room_id,))
    room = cursor.fetchone()
    return make_response(jsonify(room), 200)

@app.route('/api/rooms/account/<account_id>', methods=['GET'])
def take_rooms_by_account(account_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""
    SELECT rooms.room_id, room_name, description, train_status 
    FROM rooms, accounts, models 
    WHERE account_id = %s AND id = account_id AND models.room_id = rooms.room_id""", (account_id,))
    rooms = cursor.fetchall()
    return make_response(jsonify(rooms), 200)

@app.route('/api/rooms/room/', methods=['GET'])
def take_rooms_by_room():
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT room_id FROM rooms WHERE room_name = %s AND description = %s AND account_id = %s', 
                   (data['room_name'], data['description'], data['account_id']))
    room = cursor.fetchone()
    return make_response(jsonify(room), 200)

@app.route('/api/rooms/', methods=['POST'])
def add_room():
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('INSERT INTO rooms (room_name, description, account_id) VALUES (%s, %s, %s)', 
                   (data['room_name'], data['description'], data['account_id']))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Create room successfully!'}), 201)

@app.route('/api/rooms/<room_id>', methods=['PUT'])
def change_room(room_id):
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('UPDATE rooms SET room_name = %s, description = %s WHERE room_id = %s', 
                   ( data['room_name'], data['description'], room_id))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Edit room successfully!'}), 200)

@app.route('/api/rooms/<room_id>', methods=['DELETE'])
def erase_room(room_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('DELETE FROM rooms WHERE room_id = %s', (room_id,))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Delete room successfully!'}), 200)


# join_rooms
@app.route('/api/join_rooms/<room_id>', methods=['GET'])
def take_join_rooms(room_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""
    SELECT id, std_id, fname, lname, check_status, join_room_id
    FROM accounts AS a, join_rooms AS jr, rooms AS r 
    WHERE id = jr.account_id AND jr.room_id = r.room_id AND r.room_id = %s""", (room_id,))
    join_rooms = cursor.fetchall()
    return make_response(jsonify(join_rooms), 200)

@app.route('/api/join_rooms/<room_id>/<account_id>', methods=['GET'])
def take_join_rooms_by_account(room_id, account_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute("""
    SELECT id, std_id, fname, lname, check_status, join_room_id
    FROM accounts AS a, join_rooms AS jr, rooms AS r 
    WHERE id=jr.account_id AND jr.room_id = r.room_id AND r.room_id = %s AND id = %s""", (room_id, account_id))
    join_rooms = cursor.fetchone()
    return make_response(jsonify(join_rooms), 200)

@app.route('/api/join_rooms/', methods=['POST'])
def add_join_room():
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('INSERT INTO join_rooms (check_status, account_id, room_id) VALUES (%s, %s, %s)', 
                   (data['check_status'], data['account_id'], data['room_id']))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Join room successfully!'}), 201)

@app.route('/api/join_rooms/<room_id>/<account_id>', methods=['PUT'])
def change_join_room(room_id, account_id):
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('UPDATE join_rooms SET check_status = %s WHERE room_id = %s AND account_id = %s', 
                   (data['check_status'], room_id, account_id))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Update status successfully!'}), 200)

@app.route('/api/join_rooms/<room_id>/<account_id>', methods=['DELETE'])
def erase_join_room(room_id, account_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('DELETE FROM join_rooms WHERE room_id = %s AND account_id = %s', (room_id, account_id))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Kick participant successfully!'}), 200)


# models
@app.route('/api/models/<room_id>', methods=['GET'])
def take_model(room_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT model_name FROM models WHERE room_id = %s', (room_id,))
    model = cursor.fetchone()
    return make_response(jsonify(model), 200)

@app.route('/api/models/', methods=['POST'])
def add_model():
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('INSERT INTO models (model_name, train_status, room_id) VALUES (%s, %s, %s)', 
                   (data['model_name'], data['train_status'], data['room_id']))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Create model successfully!'}), 201)

@app.route('/api/models/<room_id>', methods=['PUT'])
def change_model(room_id):
    data = request.get_json()
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('UPDATE models SET model_name = %s, train_status = %s WHERE room_id = %s', 
                   (data['model_name'], data['train_status'], room_id))
    mysql.connection.commit()
    return make_response(jsonify({'message' : 'Train model successfully!'}), 200)


# app
@app.route('/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        data = {
            'username' : request.form['username'],
            'password' : request.form['password']
        }
        # Check if account exists using MySQL
        response = requests.get(f"{my_url}/api/accounts/login/", json = data)
        # Fetch one record and return result
        account = response.json()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)

# http://localhost:5000/python/logout - this will be the logout page
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        data = {
            'username' : request.form['username'], 
            'password' : request.form['password'], 
            'email' : request.form['email'], 
            'std_id' : request.form['std_id'], 
            'fname' : request.form['fname'], 
            'lname' : request.form['lname']
        }
        # Check if account exists using MySQL
        response = requests.get(f"{my_url}/api/accounts/username/{data['username']}")
        account = response.json()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', data['email']):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', data['username']):
            msg = 'Username must contain only characters and numbers!'
        elif not data['username'] or not data['password'] or not data['email']:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            response = requests.post(f"{my_url}/api/accounts/", json=data)
            msg = response.json()['message']
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/home - this will be the home page, only accessible for loggedin users
@app.route('/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        response = requests.get(f"{my_url}/api/rooms/")
        rooms = response.json()
        rooms = [(row['room_id'], row['room_name'], row['description']) for row in rooms]
        response = requests.get(f"{my_url}/api/accounts/{session['id']}")
        account = response.json()
        fname = account['fname']
        return render_template('home.html', rooms=rooms, id=session['id'], username=session['username'], fname=fname)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/profile - this will be the profile page, only accessible for loggedin users
@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        response = requests.get(f"{my_url}/api/accounts/{session['id']}")
        account = response.json()
        response = requests.get(f"{my_url}/api/signatures/{session['id']}")
        images = response.json()
        images = [tuple(row.values()) for row in images]
        # Show the profile page with account info
        return render_template('profile.html', account=account, images = images)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# go to edit profile page
@app.route('/edit/<username>' , methods=['POST', 'GET'])
def edit_profile(username):
    if request.method == 'GET':
        response = requests.get(f"{my_url}/api/accounts/{session['id']}")
        account = response.json()
        print(account)
        return render_template('edit.html', account = account)
    
    elif request.method == 'POST':
        data = {
            'std_id' : request.form['std_id'],
            'fname' : request.form['fname'],
            'lname' : request.form['lname']
        }
        response = requests.put(f"{my_url}/api/accounts/{session['id']}", json=data)
        return redirect(url_for('profile'))

#go to upload image
@app.route('/upload/<id>' , methods=['POST', 'GET'])
def upload_image(id):
    if request.method == 'GET':
        return  render_template('upload.html', id = id) 
    
    elif request.method == 'POST':
        image_files = request.files.getlist('file[]')
        for image_file in image_files:
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)              
                image_file.save(image_path)
                image = open(image_path, 'rb').read()
                image = b64encode(image).decode('utf-8')
                print(type(image))
                print(image)
                data = {
                    'signature_image' : image,
                    'account_id' : id
                }
                response = requests.post(f"{my_url}/api/signatures/", json=data)
                os.remove(image_path)
        return redirect(url_for('profile'))

#     
@app.route('/profile/delete/<signature_id>' , methods=['DELETE','GET'])
def manage_image(signature_id):
    if 'loggedin' in session:
        response = requests.delete(f"{my_url}/api/signatures/{signature_id}")
        return redirect(url_for('profile'))

@app.route('/home/createroom' , methods=['POST', 'GET'])
def createroom():
    if request.method == 'GET':
        return render_template('createroom.html')
    
    elif request.method == 'POST':
        data = {
            'room_name' : request.form['room_name'],
            'description' : request.form['description'],
            'account_id' : session['id']
        }
        response = requests.post(f"{my_url}/api/rooms/", json=data)
        response = requests.get(f"{my_url}/api/rooms/room", json=data)
        room = response.json()
        data = {
            'model_name' : 'signet_model',
            'train_status' : 'untrained',
            'room_id' : room['room_id']
        }
        response = requests.post(f"{my_url}/api/models/", json=data)
        flash('Success')
        return redirect(url_for('home'))

@app.route('/home/manageroom/' , methods=['POST', 'GET'])
def manageroom():
    if 'loggedin' in session:
        response = requests.get(f"{my_url}/api/rooms/account/{session['id']}")
        myrooms = response.json()
        myrooms = [(row['room_id'], row['room_name'], row['description'], row['train_status']) for row in myrooms]
        return render_template('manageroom.html' , myrooms=myrooms )

@app.route('/home/manageroom/delete/<room_id>' , methods=['POST', 'GET'])
def deleteRoom(room_id):
    if 'loggedin' in session:
        response = requests.get(f"{my_url}/api/models/{room_id}")
        model_data = response.json()
        model_name = model_data['model_name']
        if model_name != 'signet_model':
            model_path = os.path.join(app_dir,'static','models', model_name+".h5")
            os.remove(model_path)
        response = requests.delete(f"{my_url}/api/rooms/{room_id}")
        return redirect(url_for('manageroom'))

@app.route('/home/manageroom/editroom/<room_id>' , methods=['POST', 'GET'])
def editroom(room_id):
    if request.method == 'GET':
        response = requests.get(f"{my_url}/api/rooms/{room_id}")
        room = response.json()
        response = requests.get(f"{my_url}/api/join_rooms/{room_id}")
        acc_join = response.json()
        acc_join = [(row['id'], row['std_id'], row['fname'], row['lname']) for row in acc_join]
        return render_template('editroom.html', room=room, acc_join=acc_join)
    
    elif request.method == 'POST':
        data = {
            'room_name' : request.form['room_title'],
            'description' : request.form['description']
        }
        response = requests.put(f"{my_url}/api/rooms/{room_id}", json=data)
        flash("Update Complate !")
        return redirect(url_for('manageroom'))
    
@app.route('/home/manageroom/editroom/delete/<room_id>/<id>' , methods=['POST', 'GET'])
def kick_user(room_id, id):
    if 'loggedin' in session:
        response = requests.delete(f"{my_url}/api/join_rooms/{room_id}/{id}")
        return redirect(url_for('editroom'))

@app.route('/home/room/<room_id>', methods=['POST', 'GET'])
def viewroom(room_id):   
  if 'loggedin' in session:
        response = requests.get(f"{my_url}/api/rooms/{room_id}")
        inforoom = response.json()
        response = requests.get(f"{my_url}/api/join_rooms/{room_id}")
        acc_join = response.json()
        acc_join = [(row['std_id'], row['fname'], row['lname'], row['check_status']) for row in acc_join]
        return render_template('room.html', inforoom=inforoom, acc_join=acc_join)

@app.route('/home/room/join/<room_id>', methods=['POST', 'GET'])
def joinroom(room_id):
    if 'loggedin' in session:
        response = requests.get(f"{my_url}/api/join_rooms/{room_id}/{session['id']}")
        checkJoin = response.json()
        if checkJoin == None:
            data = {
                'check_status' : 'ยังไม่ตรวจสอบ', 
                'account_id' : session['id'],
                'room_id' : room_id
            }
            response = requests.post(f"{my_url}/api/join_rooms/", json=data)  
        return redirect(url_for('viewroom', room_id=room_id))

@app.route('/home/room/leave/<room_id>', methods=['POST', 'GET'])
def leaveroom(room_id):
    if 'loggedin' in session:
        response = requests.get(f"{my_url}/api/join_rooms/{room_id}/{session['id']}")
        checkJoin = response.json()              
        if checkJoin == None:
            msg = 'ยังไม่ได้ทำการเข้าห้อง'
        elif  checkJoin != None:
            response = requests.delete(f"{my_url}/api/join_rooms/{room_id}/{session['id']}")
            msg = 'Leave Successfuly'
        return redirect(url_for('viewroom', room_id=room_id))

@app.route('/home/room/trainmodel/<room_id>', methods=['POST', 'GET'])
def trainmodel(room_id):
    if 'loggedin' in session:
        response = requests.get(f"{my_url}/api/join_rooms/{room_id}")
        acc_join = response.json()
        df = pd.DataFrame(acc_join)
        batch_size = int(np.ceil(df.shape[0]*0.75))
        n_iter = 3*batch_size
        model = get_model(room_id)

        for i in range(1, n_iter+1):
            inputs, targets = get_batch(batch_size, room_id)
            loss = model.train_on_batch(inputs, targets)
        
        new_model_name = f"model_room_{room_id}"
        new_model_path = os.path.join(app_dir,'static','models', new_model_name+".h5")
        model.save_weights(new_model_path)

        data = {
            'model_name' : new_model_name,
            'train_status' : 'trained'
        }
        response = requests.put(f"{my_url}/api/models/{room_id}", json=data)
        msg = 'trained'
        return redirect(url_for('manageroom'))

@app.route('/home/room/recognition/<room_id>', methods=['POST', 'GET'])
def predict_recognition(room_id):
    if request.method == 'GET':
        response = requests.get(f"{my_url}/api/rooms/{room_id}")
        inforoom = response.json()
        return render_template('recognition.html',inforoom=inforoom)
    
    elif request.method == 'POST': 
        image_files = request.files.getlist('file[]')
        image_file = image_files[0]        
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)              
            image_file.save(image_path)
            image = img_pre.imread(image_path)
            os.remove(image_path)

            response = requests.get(f"{my_url}/api/signatures/room/{room_id}")
            myresult = response.json()
            df = pd.DataFrame(myresult)
            signers_id = df['id'].unique()
            n_signers = len(signers_id)

            support_set = []
            for category in range(n_signers):
                support_images = df[df['id'] == signers_id[category]]['signature_image']
                support_image = img_pre.imread(io.BytesIO(b64decode(rng.choice(support_images))))
                support_set.append(support_image)
            
            main_set = np.asarray([image]*n_signers)
            support_set = np.stack(support_set)
            pairs = [main_set, support_set]

            model = get_model(room_id)

            scores = model.predict(pairs)
            scores = np.array(scores)*10
            probs = np.exp(-scores) / np.sum(np.exp(-scores))

            predict_idx = np.argmax(probs)
            predict_id = signers_id[predict_idx]

            response = requests.get(f"{my_url}/api/accounts/{predict_id}")
            pre_acc = response.json()
            std_id = pre_acc['std_id']
            fname = pre_acc['fname']
            lname = pre_acc['lname']
            maxprop = np.max(probs)

            response = requests.get(f"{my_url}/api/rooms/{room_id}")
            inforoom = response.json()
            return render_template('recognition.html', std_id=std_id, fname=fname, lname=lname, maxprop=maxprop, inforoom=inforoom)
        else:
            return redirect(url_for('predict_recognition', room_id=room_id))

@app.route('/home/room/verification/<room_id>', methods=['POST', 'GET'])
def predict_verification(room_id):
    if request.method == 'GET':
        response = requests.get(f"{my_url}/api/join_rooms/{room_id}/{session['id']}")
        checkJoin = response.json()
        if checkJoin == None:
            return redirect(url_for('viewroom', room_id=room_id))
        response = requests.get(f"{my_url}/api/rooms/{room_id}")
        inforoom = response.json()
        return render_template('verification.html',inforoom=inforoom)

    elif request.method == 'POST': 
        image_files = request.files.getlist('file[]')
        std_id = request.form['std_id']
        image_file = image_files[0]        
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)              
            image_file.save(image_path)
            image = img_pre.imread(image_path)
            os.remove(image_path)

            response = requests.get(f"{my_url}/api/signatures/std_id/{std_id}")
            myresult = response.json()
            df = pd.DataFrame(myresult)
            if df.shape[0] == 0:
                response = requests.get(f"{my_url}/api/rooms/{room_id}")
                inforoom = response.json()
                return render_template('verification.html', inforoom=inforoom)

            support_set = []
            support_images = df['signature_image']
            n_support_images = len(support_images)

            for support_image in support_images:
                support_set.append(img_pre.imread(io.BytesIO(b64decode(support_image))))
            
            main_set = np.asarray([image]*n_support_images)
            support_set = np.stack(support_set)
            pairs = [main_set, support_set]

            model = get_model(room_id)
            threshold = 0.35
            
            scores = model.predict(pairs)
            target_pred = np.where(scores < threshold, 1, 0).reshape(-1)
            target_pred = pd.Series(target_pred)
            most_target_pred = target_pred.mode().values[0]

            if most_target_pred == 0:
                predict_genre = "เป็นลายเซ็นลอกเลียนแบบ"
                check_status = 'ไม่ผ่านการตรวจสอบ'
            elif most_target_pred == 1:
                predict_genre = "เป็นลายเซ็นของจริง"
                check_status = 'ผ่านการตรวจสอบ'
                
            response = requests.get(f"{my_url}/api/accounts/std_id/{std_id}")
            account = response.json()
            data = {'check_status' : check_status}
            response = requests.put(f"{my_url}/api/join_rooms/{room_id}/{account['id']}", json=data)
            response = requests.get(f"{my_url}/api/rooms/{room_id}")
            inforoom = response.json()
            return  render_template('verification.html', inforoom=inforoom, predict_genre=predict_genre)
        else:
            return  redirect(url_for('predict_verification', room_id=room_id))


@app.route('/home/room/export/<room_id>', methods=['POST', 'GET'])
def export_file(room_id):
    if request.method == 'GET':
        response = requests.get(f"{my_url}/api/join_rooms/{room_id}")
        acc_join = response.json()

        wb = Workbook()
        ws = wb.active

        # Set the column headers
        ws.append(['Std Id', 'First Name', 'Last Name', 'Status'])

        # Add the data to the worksheet
        for row in acc_join:
            ws.append([row['std_id'], row['fname'], row['lname'], row['check_status']])

        # Write the workbook to a BytesIO object
        file = io.BytesIO()
        wb.save(file)

        # Seek to the beginning of the BytesIO object
        file.seek(0)
        

        # Create a response object with the workbook data
        return Response(file.read(), mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition":"attachment;filename=verification_report.xlsx"})

if __name__ == "__main__":
    app.run(debug=True)

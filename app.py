import imageprocess
import predictor
import pickle
from flask import Flask
import flask
from flask import request
from flask import render_template
import numpy as np
import cv2
from werkzeug.utils import secure_filename



app = Flask(__name__)
applemodelpath = 'models/Applemodel_V1.sav'
apple_model = pickle.load(open(applemodelpath, 'rb'))

cornmodelpath = 'models/cornmodel_V1.sav'
corn_model = pickle.load(open(cornmodelpath, 'rb'))

grapesmodelpath = 'models/grapesmodel_V1.sav'
grapes_model = pickle.load(open(grapesmodelpath, 'rb'))

potatomodelpath = 'models/potatomodel_V1.sav'
potato_model = pickle.load(open(potatomodelpath, 'rb'))

tomatomodelpath = 'models/Tomatomodel_V1.sav'
tomato_model = pickle.load(open(tomatomodelpath, 'rb'))


@app.route("/")
def home():
	version = "1.1"
	return render_template('index.html',version1=version)

@app.route("/predict", methods = ['GET', 'POST'])
def submit():
	imagefile = flask.request.files["data_file"].read()
	file = request.files['data_file']
	filename = secure_filename(file.filename)
	print(filename)
	dname = request.form.get('Name')
	dname = str(dname)
	response = dname[0]
	npimg = np.frombuffer(imagefile, np.uint8)
	# convert numpy array to image
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

	f_vector = imageprocess.feature_extractor(img)

	if response=='n':
		res = "Please select the appropriate plant from the list"
		return '<h1>'+res+'</h1>'

	if response=='a':
		p_vector = [f_vector['area'],f_vector['perimeter'],f_vector['red_mean'],f_vector['blue_mean'],f_vector['f2'],f_vector['green_std'],
		f_vector['f4'],f_vector['f6'],f_vector['f7']]

		res = predictor.apple_p(p_vector,apple_model)
		var="static/"+filename
		return f'<img src={var} > <br><h1> The Disease is  Identified is :  {res}</h1>'

	if response=='c':
		p_vector = [f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'], f_vector['blue_std'],
		f_vector['f7'], f_vector['f8']]

		res = predictor.corn_p(p_vector,corn_model)
		var="static/"+filename
		return f'<img src={var} > <br><h1> The Disease is  Identified is :  {res}</h1>'

	if response=='g':
		p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['red_mean'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'],
	       f_vector['red_std'], f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f6'], f_vector['f7'], f_vector['f8']]

		res = predictor.grapes_p(p_vector,grapes_model)
		var="static/"+filename
		return f'<img src={var} > <br><h1> The Disease is  Identified is :  {res}</h1>'

	if response=='p':
		p_vector = [f_vector['area'], f_vector['perimeter'], f_vector['green_mean'], f_vector['blue_mean'], f_vector['f2'], f_vector['red_std'],
	       f_vector['green_std'], f_vector['blue_std'], f_vector['f4'], f_vector['f5'], f_vector['f7'], f_vector['f8']]

		res = predictor.potato_p(p_vector,potato_model)
		var="static/"+filename
		return f'<img src={var} > <br><h1> The Disease is  Identified is :  {res}</h1>'

	if response=='t':
		del f_vector["f1"]
		p_vector = list(f_vector.values())

		res = predictor.tomato_p(p_vector,tomato_model)
		var="static/"+filename
		return f'<img src={var} > <br><h1> The Disease is  Identified is :  {res}</h1>'

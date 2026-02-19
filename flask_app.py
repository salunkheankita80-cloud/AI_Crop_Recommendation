from flask import Flask, render_template, request
import pickle


app=Flask(__name__)
with open ('crop_model.pkl','rb') as f:
    model= pickle.load(f)

@app.route('/' ,methods=['GET','POST'])
def crop_recomdation_view():
    if request.method =='GET':
        return render_template('crop_recomadation.html')

    elif request.method == 'POST':
        Nitrogen = float(request.form['N'])
        Phosphours = float(request.form['P'])
        Potassium = float(request.form['K'])
        Temperature  = float(request.form['T'])
        humidity = float(request.form['h'])
        pH  = float(request.form['ph'])
        Rainfall = float(request.form['R'])
        y_pred = model.predict([[Nitrogen,Phosphours, Potassium,Temperature ,humidity,pH ,Rainfall]])
        y_pred =( y_pred[0])

        return render_template('prediction.html', ans=y_pred)


if __name__ == '__main__':
    app.run(debug=True)


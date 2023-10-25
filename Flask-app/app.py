import pandas as pd
from flask import Flask, render_template, url_for, request
import pickle

app = Flask(__name__)

def predict(tahun):
    datet=[]

    while tahun > 2022:
        datet.append(tahun)
        tahun=tahun-1
    datet=pd.DataFrame(datet)
    datet.columns=['date']
    datet['date']=pd.to_datetime(datet['date'],format='%Y')
    datet.set_index('date',inplace=True)
    datet.sort_index(inplace=True)
    pickled_model = pickle.load(open('model.pkl', 'rb'))
    y_pred=pickled_model.forecast(steps=len(datet))
    datet.reset_index(inplace=True)
    return y_pred,datet

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=='GET':
        return render_template('index.html')
    elif request.method=='POST':
        tahun= int(request.form["tahun"])
        y_pred,datet = predict(tahun)
        datet['date']=datet['date'].astype(str)
        datet=datet['date'].values.tolist()
        #take four only four first char datet
        datet=[i[0:4] for i in datet]
        y_pred=y_pred.values
        return render_template('Result.html', 
                                data=y_pred,
                                tahun=datet)

if __name__ == '__main__':
    app.run(debug=True)
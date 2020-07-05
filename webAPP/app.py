from flask import (Flask, render_template,request)
from modelModule import cnnTF
app = Flask(__name__)


predictor = cnnTF()
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        dta = request.args.get('imgdata')
        if dta is not None:
            dta = dta[22:]
            print(type(dta))
            prediction = predictor.b64Pred(dta)
            print("\n", "PREDICTION IS", prediction)
            if prediction == 0:
                dta = "Wearing mask"
            else:
                dta = "Not wearing mask"
        return render_template(
            'index.html',
            dta = dta
        )
        
    else:
        return render_template(
            'index.html'
        )
    

if __name__ == '__main__':
    app.run(debug=False)



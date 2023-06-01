from flask import Flask

app = Flask(__name__)

#Test

@app.route("/members")
def members():
    return {"members": ["M1","M2","M4"]}


# @app.route('/')
# def index():
#     return 


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
from flask import Flask

app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hey Vijay, how are you?'

if __name__ == '__main__':
    app.run()
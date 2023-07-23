from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

app.template_folder = "./"

@app.route('/')
def index():
    return render_template('practice_2_index.html')


@app.route('/<code>', methods=['GET'])
def check_(code):
    Cipher = '路由装饰器'
    if request.method == 'GET' and code == Cipher:
        return '使用GET方法，口令正确'


@app.route('/check', methods=['POST'])
def check():
    Cipher = '路由装饰器'
    cipher = request.form.get('cipher')
    if request.method == 'POST' and cipher == Cipher:
        return redirect(url_for('success'))


@app.route('/success')
def success():
    return '暗号对接成功！'


if __name__ == '__main__':
    app.run()

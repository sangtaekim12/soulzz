#!/usr/bin/env python3
"""
간단한 인증 시스템을 추가하는 스크립트
"""

def add_simple_auth_to_app():
    """Flask 앱에 간단한 인증 추가"""
    
    auth_code = """
from functools import wraps
from flask import request, redirect, url_for, session, render_template_string

# 비밀번호 설정 (실제 환경에서는 환경변수나 설정파일 사용)
SECRET_PASSWORD = "tourism2024"  # 원하는 비밀번호로 변경
app.secret_key = "your-secret-key-here"  # 세션용 비밀키

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == SECRET_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="잘못된 비밀번호입니다.")
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    session.pop('authenticated', None)
    return redirect(url_for('login'))

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>관광 챗봇 로그인</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h4 class="text-center">🤖 관광 챗봇</h4>
                    </div>
                    <div class="card-body">
                        <form method="post">
                            <div class="mb-3">
                                <label for="password" class="form-label">비밀번호</label>
                                <input type="password" class="form-control" name="password" required>
                            </div>
                            {% if error %}
                            <div class="alert alert-danger">{{ error }}</div>
                            {% endif %}
                            <button type="submit" class="btn btn-primary w-100">로그인</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''
"""
    
    print("🔐 인증 시스템 코드가 준비되었습니다.")
    print()
    print("인증을 추가하려면:")
    print("1. app.py 파일 상단에 위 코드를 추가")
    print("2. 기존 라우트에 @require_auth 데코레이터 추가")
    print("3. 비밀번호 변경 (기본값: tourism2024)")
    print()
    print("예시:")
    print("@app.route('/')")
    print("@require_auth")
    print("def index():")
    print()
    
    return auth_code

if __name__ == "__main__":
    auth_code = add_simple_auth_to_app()
    
    # 파일로 저장
    with open('auth_code.txt', 'w') as f:
        f.write(auth_code)
    
    print("✅ 인증 코드가 auth_code.txt 파일에 저장되었습니다.")
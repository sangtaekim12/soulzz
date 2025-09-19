#!/usr/bin/env python3
"""
Ollama 연결 설정
"""

import os

# Ollama 서버 설정
OLLAMA_CONFIG = {
    # 기본값: 로컬 서버 (현재 사용 중)
    'host': 'http://localhost:11434',
    
    # 사용자 컴퓨터 연결 시 아래 주석을 해제하고 IP 변경
    # 'host': 'http://192.168.1.100:11434',  # 사용자 컴퓨터 IP
    
    'model': 'nous-hermes2',
    'timeout': 30
}

# 환경변수로 덮어쓰기 가능
OLLAMA_CONFIG['host'] = os.getenv('OLLAMA_HOST', OLLAMA_CONFIG['host'])
OLLAMA_CONFIG['model'] = os.getenv('OLLAMA_MODEL', OLLAMA_CONFIG['model'])

def get_ollama_client():
    """Ollama 클라이언트 생성"""
    try:
        import ollama
        
        # 호스트가 localhost가 아닌 경우 원격 클라이언트 생성
        if 'localhost' not in OLLAMA_CONFIG['host'] and '127.0.0.1' not in OLLAMA_CONFIG['host']:
            client = ollama.Client(host=OLLAMA_CONFIG['host'])
            print(f"🌐 원격 Ollama 서버에 연결: {OLLAMA_CONFIG['host']}")
        else:
            client = ollama
            print(f"🏠 로컬 Ollama 서버에 연결: {OLLAMA_CONFIG['host']}")
        
        # 연결 테스트
        try:
            models = client.list()
            available_models = [m['name'] for m in models.get('models', [])]
            
            if OLLAMA_CONFIG['model'] in available_models:
                print(f"✅ 모델 '{OLLAMA_CONFIG['model']}' 사용 가능")
                return client
            else:
                print(f"⚠️ 모델 '{OLLAMA_CONFIG['model']}' 없음. 사용 가능: {available_models}")
                return None
                
        except Exception as e:
            print(f"❌ Ollama 연결 실패: {e}")
            return None
            
    except ImportError:
        print("❌ ollama 패키지가 설치되지 않음")
        return None

def test_connection():
    """연결 테스트"""
    print("🔍 Ollama 연결 테스트")
    print("=" * 40)
    
    client = get_ollama_client()
    
    if client:
        try:
            # 간단한 테스트 메시지
            response = client.chat(
                model=OLLAMA_CONFIG['model'],
                messages=[{'role': 'user', 'content': '안녕하세요'}]
            )
            
            print("✅ 연결 및 응답 테스트 성공!")
            print(f"🤖 응답: {response['message']['content'][:50]}...")
            return True
            
        except Exception as e:
            print(f"❌ 응답 테스트 실패: {e}")
            return False
    else:
        print("❌ 클라이언트 생성 실패")
        return False

if __name__ == "__main__":
    print("⚙️ 현재 Ollama 설정:")
    print(f"  호스트: {OLLAMA_CONFIG['host']}")
    print(f"  모델: {OLLAMA_CONFIG['model']}")
    print()
    
    test_connection()
    
    print("\n📝 사용자 컴퓨터 연결 방법:")
    print("1. 사용자 컴퓨터에서 Ollama 설치 및 실행")
    print("2. 외부 접근 허용: set OLLAMA_HOST=0.0.0.0:11434")
    print("3. config.py에서 host를 사용자 컴퓨터 IP로 변경")
    print("4. 방화벽에서 11434 포트 허용")
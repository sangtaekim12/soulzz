#!/usr/bin/env python3
"""
사용자 컴퓨터의 Ollama 서비스에 연결하는 설정
"""

import ollama
import os

def configure_remote_ollama():
    """원격 Ollama 연결 설정"""
    
    print("🔧 사용자 컴퓨터 Ollama 연결 설정")
    print("=" * 50)
    
    print("\n📋 필요한 사전 작업:")
    print("1. 사용자 컴퓨터에 Ollama 설치")
    print("2. Ollama 서비스 외부 접근 허용")
    print("3. 네트워크 방화벽 설정")
    
    print("\n🖥️ 사용자 컴퓨터에서 실행할 명령:")
    print("# Ollama 설치 (Windows)")
    print("winget install Ollama.Ollama")
    print()
    print("# 또는 https://ollama.com/download 에서 직접 다운로드")
    print()
    print("# 모델 다운로드")
    print("ollama pull nous-hermes2")
    print()
    print("# 외부 접근 허용 (환경변수 설정)")
    print("set OLLAMA_HOST=0.0.0.0:11434")
    print("ollama serve")
    
    print("\n🌐 연결 설정 코드:")
    
    config_code = '''
# simple_rag.py 파일 수정
def generate_answer(self, question):
    try:
        # 사용자 컴퓨터 IP로 연결 (예시)
        import ollama
        
        # 원격 Ollama 서버 설정
        client = ollama.Client(host='http://사용자컴퓨터IP:11434')
        
        retrieved_docs, sources = self.query(question)
        if not retrieved_docs:
            return "관련 정보를 찾을 수 없습니다.", []
        
        context = "\\n".join(retrieved_docs[:2])
        prompt = f"""
다음 자료를 바탕으로 한국어로 답변하세요:

{context}

질문: {question}
답변:
"""
        
        response = client.chat(
            model="nous-hermes2",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response['message']['content'], sources[:3]
        
    except Exception as e:
        # 폴백 시스템 사용
        return self.generate_simple_answer(question)
'''
    
    print(config_code)
    
    print("\n⚠️ 주의사항:")
    print("- 사용자 컴퓨터의 IP 주소 필요")
    print("- 방화벽에서 11434 포트 열기")
    print("- Ollama 서비스가 계속 실행되어야 함")
    print("- 네트워크 연결 안정성 필요")

def test_remote_connection(host_ip):
    """원격 Ollama 연결 테스트"""
    try:
        client = ollama.Client(host=f'http://{host_ip}:11434')
        models = client.list()
        print(f"✅ {host_ip}:11434 연결 성공!")
        print(f"📋 사용 가능한 모델: {len(models.get('models', []))}개")
        return True
    except Exception as e:
        print(f"❌ {host_ip}:11434 연결 실패: {e}")
        return False

if __name__ == "__main__":
    configure_remote_ollama()
    
    # 테스트할 IP가 있다면
    test_ip = input("\n테스트할 사용자 컴퓨터 IP (선택사항, 엔터로 건너뛰기): ").strip()
    if test_ip:
        test_remote_connection(test_ip)
#!/usr/bin/env python3
"""
OpenAI API를 사용한 고품질 LLM 연동
"""

def setup_openai_integration():
    """OpenAI API 연동 설정"""
    
    print("🤖 OpenAI API 연동 설정")
    print("=" * 40)
    
    setup_code = '''
# requirements.txt에 추가
openai>=1.0.0

# .env 파일 생성 (API 키 보안)
OPENAI_API_KEY=your-api-key-here

# simple_rag.py 수정
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class SimpleTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.1):
        # 기존 코드...
        self.openai_client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
    
    def generate_answer_with_openai(self, question):
        """OpenAI API를 사용한 답변 생성"""
        try:
            retrieved_docs, sources = self.query(question)
            
            if not retrieved_docs:
                return "관련 정보를 찾을 수 없습니다.", []
            
            context = "\\n".join(retrieved_docs[:3])
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 또는 gpt-3.5-turbo (더 저렴)
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 관광 전문 도우미입니다. 주어진 자료를 바탕으로 정확하고 도움이 되는 한국어 답변을 제공하세요."
                    },
                    {
                        "role": "user", 
                        "content": f"""
다음 관광 자료를 참고하여 질문에 답해주세요:

자료:
{context}

질문: {question}

답변 요구사항:
- 제공된 자료만을 근거로 답변
- 자료에 없는 내용은 추측하지 말 것
- 친근하고 전문적인 톤 사용
- 한국어로 200자 내외로 간결하게 작성
"""
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            return answer, sources[:3]
            
        except Exception as e:
            logger.error(f"OpenAI API 오류: {e}")
            return self.generate_simple_answer(question)
    
    def generate_answer(self, question):
        """메인 답변 생성 함수 (우선순위: OpenAI > Ollama > 폴백)"""
        # 1순위: OpenAI API
        if os.getenv('OPENAI_API_KEY'):
            return self.generate_answer_with_openai(question)
        
        # 2순위: Ollama (기존 코드)
        try:
            import ollama
            # 기존 Ollama 코드...
        except:
            pass
        
        # 3순위: 폴백 시스템
        return self.generate_simple_answer(question)
'''
    
    print("📋 설정 단계:")
    print("1. OpenAI 계정 생성 (https://openai.com)")
    print("2. API 키 발급")
    print("3. 환경변수 설정")
    print("4. 코드 통합")
    
    print("\n💰 비용 정보:")
    print("- GPT-4o-mini: 입력 $0.15/1M토큰, 출력 $0.6/1M토큰")
    print("- GPT-3.5-turbo: 입력 $0.5/1M토큰, 출력 $1.5/1M토큰")
    print("- 일반적인 질답: 약 $0.001-0.01 per 질문")
    
    print("\n✅ 장점:")
    print("- 높은 품질의 답변")
    print("- 안정적인 서비스")
    print("- 빠른 응답속도")
    print("- 별도 하드웨어 불필요")
    
    return setup_code

def create_openai_env_template():
    """OpenAI 환경변수 템플릿 생성"""
    env_content = """# OpenAI API 설정
OPENAI_API_KEY=sk-your-actual-api-key-here

# 선택사항: 모델 설정
OPENAI_MODEL=gpt-4o-mini

# 선택사항: 답변 길이 제한
MAX_TOKENS=500
"""
    
    with open('.env.template', 'w') as f:
        f.write(env_content)
    
    print("✅ .env.template 파일이 생성되었습니다.")
    print("📝 실제 API 키를 입력한 후 .env로 이름을 변경하세요.")

if __name__ == "__main__":
    setup_code = setup_openai_integration()
    create_openai_env_template()
    
    print("\n" + "="*50)
    print("🚀 통합 코드가 준비되었습니다!")
    print("원하는 방식을 선택하여 적용하세요.")
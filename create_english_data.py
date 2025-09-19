#!/usr/bin/env python3
"""
영어로 된 샘플 관광 데이터를 생성하는 스크립트
"""

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_english_sample_pdfs():
    """영어 샘플 관광 정보 PDF 파일들을 생성"""
    
    # data 폴더 생성
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 기존 PDF 파일 삭제
    for f in os.listdir(data_dir):
        if f.endswith('.pdf'):
            os.remove(os.path.join(data_dir, f))
    
    # 영어 샘플 관광 정보 데이터
    sample_data = {
        "seoul_tourism.pdf": [
            "Seoul Tourism Guide",
            "",
            "Gyeongbokgung Palace is the main royal palace of the Joseon dynasty, built in 1395.",
            "The palace holds daily changing of the guard ceremonies, and entry is free when wearing hanbok.",
            "",
            "Namsan Seoul Tower is Seoul's representative landmark.",
            "From Namsan Tower, you can see the entire Seoul cityscape, especially beautiful at night.",
            "",
            "Myeongdong is Seoul's representative shopping district.",
            "Myeongdong has various brand stores, cosmetics shops, and restaurants concentrated in one area.",
            "",
            "Insadong is famous as a street of traditional culture.",
            "In Insadong, you can enjoy traditional tea at tea houses and buy Korean traditional crafts.",
        ],
        
        "busan_travel.pdf": [
            "Busan Travel Guide",
            "",
            "Haeundae Beach is Busan's representative tourist destination.",
            "Haeundae is famous for its wide sandy beach and clean sea, popular as a summer resort.",
            "",
            "Gwangalli Beach is famous for the night view of Gwangan Bridge.",
            "At Gwangalli, you can taste various seafood dishes and there are many cafes and bars.",
            "",
            "Gamcheon Culture Village is called 'Korea's Machu Picchu'.",
            "Gamcheon Culture Village boasts unique scenery with colorful houses arranged in steps.",
            "",
            "Jagalchi Market is Busan's largest fish market.",
            "At Jagalchi Market, you can taste fresh sashimi and various seafood at affordable prices.",
        ],
        
        "jeju_island.pdf": [
            "Jeju Island Attractions Guide",
            "",
            "Hallasan Mountain is the symbol of Jeju Island and the highest peak in South Korea.",
            "Climbing Hallasan offers beautiful scenery in different seasons, with the Baengnokdam crater trail being popular.",
            "",
            "Seongsan Ilchulbong is a volcanic formation located in eastern Jeju.",
            "The sunrise view from Seongsan Ilchulbong is one of Jeju's representative tourist highlights.",
            "",
            "Udo Island is a small island in the eastern sea of Jeju.",
            "On Udo, you can enjoy beautiful beaches, fresh seafood, and famous peanut ice cream.",
            "",
            "Jungmun Resort is a representative resort area in southern Jeju.",
            "Jungmun area has various resorts, golf courses, and theme parks.",
        ],
        
        "korean_cuisine.pdf": [
            "Korean Food Tourism Guide",
            "",
            "Kimchi is Korea's representative fermented food.",
            "Kimchi is made with various vegetables like cabbage, radish, and cucumber, with different tastes and types by region.",
            "",
            "Bulgogi is Korea's traditional grilled dish.",
            "Bulgogi is marinated thin sliced beef grilled in sweet sauce, popular among foreigners.",
            "",
            "Bibimbap is a Korean dish mixing various vegetables with rice.",
            "Bibimbap has excellent nutritional balance and is characterized by mixing with gochujang.",
            "",
            "Hansik is Korea's traditional course meal.",
            "In hansik, you can taste various side dishes and seasonal ingredient dishes.",
        ],
        
        "cultural_heritage.pdf": [
            "Korean Cultural Heritage Guide",
            "",
            "Changdeokgung Palace is a Joseon dynasty palace and UNESCO World Cultural Heritage.",
            "The secret garden of Changdeokgung is acclaimed as the essence of Korean traditional landscaping in harmony with nature.",
            "",
            "Bulguksa Temple is a representative temple of the Silla period.",
            "Bulguksa has two national treasure stone pagodas called Dabotap and Seokgatap.",
            "",
            "Seokguram Grotto is a representative cultural heritage site in Gyeongju along with Bulguksa.",
            "The main Buddha statue of Seokguram is recognized as the finest sculpture work in the East.",
            "",
            "Jongmyo Shrine is a royal ancestral shrine housing the spirit tablets of Joseon dynasty kings and queens.",
            "Jongmyo ritual music is registered as UNESCO Intangible Cultural Heritage traditional court music.",
        ]
    }
    
    # PDF 파일 생성
    for filename, content in sample_data.items():
        filepath = os.path.join(data_dir, filename)
        
        # 간단한 텍스트 PDF 생성
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        
        y_position = height - 50
        line_height = 20
        
        for line in content:
            if y_position < 50:  # 새 페이지 시작
                c.showPage()
                y_position = height - 50
                
            if line.strip():  # 빈 줄이 아닌 경우
                c.drawString(50, y_position, line)
            
            y_position -= line_height
            
        c.save()
        print(f"Created: {filepath}")
    
    print(f"\nTotal {len(sample_data)} English sample PDF files created.")
    print(f"File location: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    try:
        create_english_sample_pdfs()
    except Exception as e:
        print(f"Error occurred: {e}")
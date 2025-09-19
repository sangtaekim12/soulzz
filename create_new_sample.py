#!/usr/bin/env python3
"""
새로운 관광 정보 PDF 생성 예시
"""

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_new_travel_guide():
    """새로운 관광 가이드 PDF 생성"""
    
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 새로운 관광 정보
    travel_content = {
        "korea_hidden_gems.pdf": [
            "Korea Hidden Gems Travel Guide",
            "",
            "Bukchon Hanok Village preserves traditional Korean architecture.",
            "Walk through narrow alleys between hanok houses dating back 600 years.",
            "The village offers panoramic views of Seoul and traditional tea houses.",
            "",
            "Seoraksan National Park is famous for its autumn foliage.",
            "The park features hiking trails, waterfalls, and Buddhist temples.",
            "Ulsanbawi Rock provides breathtaking views of the surrounding mountains.",
            "",
            "Boseong Tea Fields create a green carpet landscape.",
            "Visit during spring or summer for the most vibrant green scenery.",
            "The area is perfect for photography and peaceful walks.",
        ],
        
        "korean_festivals.pdf": [
            "Korean Seasonal Festivals Guide",
            "",
            "Cherry Blossom Festival occurs in spring (March-May).",
            "Best locations include Yeouido Park, Namsan Park, and Jinhae.",
            "The festivals feature food stalls, cultural performances, and night illumination.",
            "",
            "Boryeong Mud Festival is held every July on Daecheon Beach.",
            "Participants enjoy mud wrestling, mud slides, and therapeutic mud treatments.",
            "The festival attracts thousands of international visitors annually.",
            "",
            "Jinju Lantern Festival takes place in October along Namgang River.",
            "Colorful lanterns float on the water creating magical night scenes.",
            "The festival commemorates the historic Battle of Jinju Castle.",
        ],
        
        "korean_spa_culture.pdf": [
            "Korean Spa and Wellness Guide",
            "",
            "Jjimjilbangs are Korean public bathhouses offering various facilities.",
            "These 24-hour spas include hot baths, saunas, sleeping areas, and restaurants.",
            "Popular chains include Dragon Hill Spa, Siloam Sauna, and Spaland.",
            "",
            "Korean traditional medicine emphasizes natural healing methods.",
            "Herbal treatments, acupuncture, and cupping are widely available.",
            "Many spas offer traditional Korean massage techniques.",
            "",
            "Hot springs are found throughout Korea, especially in Busan and Jeju.",
            "Therapeutic waters are believed to have healing properties for skin and joints.",
            "Popular hot spring destinations include Dongrae and Dogo Onsen.",
        ]
    }
    
    # PDF 파일 생성
    for filename, content in travel_content.items():
        filepath = os.path.join(data_dir, filename)
        
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        
        y_position = height - 50
        line_height = 20
        
        for line in content:
            if y_position < 50:
                c.showPage()
                y_position = height - 50
                
            if line.strip():
                c.drawString(50, y_position, line)
            
            y_position -= line_height
            
        c.save()
        print(f"Created: {filepath}")
    
    print(f"\nNew travel guide PDFs created successfully!")
    print(f"Location: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    create_new_travel_guide()
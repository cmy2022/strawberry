"""
ç®åæµè¯èæ¬ï¼éªè¯AIæºå¨äººçåºæ¬åè½
"""

from simple_neural_ai_robot import SimpleNeuralAIBot

def test_basic_functionality():
    print("æµè¯AIæºå¨äººçåºæ¬åè½...")
    
    # åå»ºæºå¨äººå®ä¾
    bot = SimpleNeuralAIBot()
    
    # æµè¯å¯¹è¯åè½
    print("\n1. æµè¯å¯¹è¯åè½:")
    response = bot.chat("ä½ å¥½ï¼è¿æ¯ä¸ä¸ªæµè¯")
    print(response)
    
    # æµè¯æç´¢åè½
    print("\n2. æµè¯æç´¢åè½:")
    response = bot.chat("è¯·æç´¢äººå·¥æºè½çä¿¡æ¯")
    print(response)
    
    # æµè¯å³ç­åè½
    print("\n3. æµè¯å³ç­åè½:")
    response = bot.chat("æåºè¯¥éæ©åªä¸ªæ·±åº¦å­¦ä¹ æ¡æ¶ï¼")
    print(response)
    
    # æµè¯æä»¤æ§è¡åè½
    print("\n4. æµè¯æä»¤æ§è¡åè½:")
    response = bot.chat("åæéæ±å¼åä¸ä¸ªèå¤©æºå¨äºº")
    print(response)
    
    print("\nææåºæ¬åè½æµè¯å®æï¼")

if __name__ == "__main__":
    test_basic_functionality()

def apply_triage_rules(ich_volumes, has_fracture, mls_mm):
    """
    اعمال قوانین تریاژ بر اساس داکیومنت مسابقه IAAA 2026.
    
    ورودی:
    ich_volumes (dict): دیکشنری حجم خونریزی‌ها از ICHPredictor
    has_fracture (bool): خروجی FracturePredictor
    mls_mm (float): خروجی MLSPredictor
    
    خروجی:
    str: "Level 1", "Level 2", یا "Normal"
    """
    
    # --- قانون ۱: آیا بیمار در وضعیت بحرانی (Level 1) است؟ ---
    
    # 1a: آیا حجم هر نوع خونریزی بیش از ۳۰ میلی‌لیتر است؟
    if any(vol > 30 for vol in ich_volumes.values()):
        return "Level 1"
        
    # 1b: آیا حجم خونریزی Epidural (EDH) یا Subdural (SDH) بیش از ۱۵ میلی‌لیتر است؟
    if ich_volumes.get("EDH", 0) > 15 or ich_volumes.get("SDH", 0) > 15:
        return "Level 1"
        
    # 1c: آیا انحراف خط میانی (MLS) بیش از ۵ میلی‌متر است؟
    if mls_mm > 5:
        return "Level 1"
        
    # 1d: آیا خونریزی داخل بطنی (IVH) وجود دارد؟ (حتی یک قطره)
    if ich_volumes.get("IVH", 0) > 0.1: # یک آستانه کوچک برای جلوگیری از خطای عددی
        return "Level 1"
        
    # 1e: آیا شکستگی جمجمه (Fracture) وجود دارد؟
    if has_fracture:
        return "Level 1"

    # --- قانون ۲: اگر بحرانی نیست، آیا بیمار نیاز به توجه (Level 2) دارد؟ ---
    
    # 2a: آیا مجموع حجم تمام خونریزی‌ها (به جز IVH) بیشتر از ۰.۱ میلی‌لیتر است؟
    total_ich = sum(v for k, v in ich_volumes.items() if k != "IVH")
    if total_ich > 0.1:
        return "Level 2"
        
    # --- قانون ۳: اگر هیچکدام از موارد بالا نبود، بیمار نرمال است. ---
    return "Normal"
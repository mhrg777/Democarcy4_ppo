import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygetwindow as gw
import pyautogui
import cv2
import time
import os
import random
import pytesseract
import re
import difflib
from fuzzywuzzy import fuzz  # نیاز به نصب کتابخانه fuzzywuzzy

# --- Constants ---
SCALE_GENERAL      = 1
SCALE_D            = 1
MATCH_THRESHOLD    = 0.75
SKIP_THRESHOLD     = 0.85
SKIP_DI_THRESHOLD  = 0.9
REL_CORNER1        = (1267, 796)
REL_CORNER         = (1269, 55)
SLIDER_MIN_X       = 65
SLIDER_MAX_X       = 985
c2_THRESHOLD       = 0.95
POP_THRESHOLD      = 0.9
E_THRESHOLD        = 0.9

pytesseract.pytesseract.tesseract_cmd = r'D:\E\Tesseract\tesseract.exe'


class DemocracyEnv(gym.Env):
    """
    محیط Gym برای بازی Democracy 4 با کنترل UI:
    - انتخاب policy (اکشن پیوسته برای index)
    - کنترل slider (اکشن پیوسته برای magnitude)
    - skip‑logic خودکار برای رد کردن نوبت
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 templates_dir=r"C:\Users\moham\OneDrive\Desktop\Project Codes\Templates"):
        super().__init__()

        # لیست سیاست‌ها
        self.policy_names = [
    'AgricultureSubsidies', 'AirlineTax', 'AlcoholLaw', 'AlcoholTax', 'ArmedForcesWeek',
    'ArmedPolice', 'BiofuelSubsidies', 'BodyCameras', 'BorderControls', 'CapitalGainsTax',
    'CarTax', 'CCTVCameras', 'CharityTaxRelief', 'ChildBenefit', 'CitizenshipForSale',
    'CitizenshipTests', 'CleanEnergySubsidies', 'CommunityPolicing', 'CompetitionLaw', 'ConsumerRights',
    'CorporationTax', 'DeathPenalty', 'DisabilityBenefit', 'DrugEnforcement', 'FamilyPlanning',
    'FirearmLaws', 'FoodStamps', 'ForeignAid', 'GameHuntingRestrictions', 'GatedCommunities',
    'GayMarriage', 'GenderDiscriminationAct', 'GenderTransition', 'GenitalMutilationBan', 'HealthTaxCredits',
    'ImmigrationRules', 'ImportTariffs', 'IncomeTax', 'InheritanceTax', 'IntellectualProperty',
    'IntelligenceServices', 'JudicialIndependence', 'JuryTrial', 'LaborDayBankHoliday', 'LaborLaws',
    'LegalAid', 'LimitAutomatedTrading', 'MilitarySpending', 'MinimumWage', 'Narcotics',
    'NuclearFission', 'NuclearWeapons', 'PayrollTax', 'PetrolTax', 'PoliceForce',
    'PollutionControls', 'PressFreedom', 'PrisonerTagging', 'PrisonRegime', 'Prisons',
    'PrivatePrisons', 'PropertyTax', 'QuantitativeEasing', 'Recycling', 'RefugeePolicy',
    'ReligiousBanknotes', 'RetirementAge', 'RoadBuilding', 'RubberBullets', 'SalesTax',
    'SecretService', 'SocialCare', 'SpaceProgram', 'SpeedCameras', 'StateHousing',
    'StatePensions', 'SubsidizedSchoolBuses', 'TearGas', 'TobaccoTax', 'UnemployedBenefit',
    'WaterCannons', 'WitnessProtection', 'WorkSafetyLaw','TechnologyColleges', 'StatePostalService',
    'BanTobacco', 'StateHealthService', 'StateSchools', 'ScienceFunding', 'SelectiveSchooling', 'StateBroadcaster',
    'FreeSchoolMeals', 'SecularityofEducation', 'ExecutiveTermLength', 'ExecutiveTermLimit', 'AbortionLaw', 
    'ArtsSubsidies', 'FaithSchoolSubsidies' , 'FoodStandardsAgency'
     ]
        
        

        self.SLIDER_BINS = 21
        
        self.game_over = False  # وضعیت پایان بازی
        self.term_count = 0
        self.max_terms = 8 # حداکثر فرضی (می‌تواند تغییر کند)
        self.is_retired = False
        self.party_member = False
        self.term_limit_reached = False
                
        # فضای اکشن: دو مولفه‌ی گسسته
        #  0: policy index ∈ [0, N-1]
        #  1: slider bin index ∈ [0, SLIDER_BINS-1]
        self.action_space = spaces.MultiDiscrete(
            [len(self.policy_names), self.SLIDER_BINS]
        )

        # فضای مشاهده: تصویر خاکستری 84×84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )

        # FailSafe را غیرفعال می‌کنیم
        pyautogui.FAILSAFE = False
        self.templates_dir = templates_dir

        # بارگذاری تمپلیت آیکون‌ها
        self.policy_icons = {
            name: cv2.imread(os.path.join(templates_dir, f"Policy_{name}.png"),
                             cv2.IMREAD_GRAYSCALE)
            for name in self.policy_names
        }
        self.slider_tpl = cv2.imread(
            os.path.join(templates_dir, 'slider_knob.png'),
            cv2.IMREAD_GRAYSCALE
        )
        self.apply_tpl = cv2.imread(
            os.path.join(templates_dir, 'apply_changes.png'),
            cv2.IMREAD_GRAYSCALE
        )
        self.ok_tpl = cv2.imread(
            os.path.join(templates_dir, 'apply_ok.png'),
            cv2.IMREAD_GRAYSCALE
        )

        
        # بارگذاری تمپلیت‌های skip logic
        path_map = {
            'popup': 'select_button_template.png',
            'a': 'icon_a.png',   'b': 'icon_b.png',
            'c': 'icon_c.png',   'c1': 'icon_c1.png',
            'd': 'icon_d.png',   'd1': 'icon_d1.png', 'd2': 'icon_d2.png',
            'e': 'icon_e.png',   'f': 'icon_f.png',   'f1': 'icon_f1.png',
            'next': 'next_turn.png',
            'cont1': 'continue1.png', 'cont2': 'continue2.png',
            'g': 'icon_g.png','assassinated': 'assassinated.png',
            'start_game_3':'start_game_3.png'

        }
        self.skip_templates = {}
        for key, fn in path_map.items():
            img = cv2.imread(os.path.join(templates_dir, fn), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                scale = SCALE_D if key in ['d','d1','d2'] else SCALE_GENERAL
                tpl = cv2.resize(img, (0,0), fx=scale, fy=scale)
                self.skip_templates[key] = (tpl, tpl.shape[::-1], scale)

        # پیدا کردن پنجره بازی
        wins = gw.getWindowsWithTitle('Democracy 4')
        if not wins:
            raise RuntimeError("Game window not found")
        self.win = wins[0]
        self.win.activate()
        time.sleep(1)
        
        # پارامترهای داخلی
        self.CONFIDENCE = MATCH_THRESHOLD
        self.current_policies = 0
        self.target_policies = random.randint(2, 5)

        self.reward_tpl = cv2.imread(os.path.join(templates_dir, 'reward.png'), cv2.IMREAD_GRAYSCALE)
        self.total_reward = 0 
        term_templates = {
        'term_limit': 'term_limit_message.png',
        'continue_as_party': 'continue_as_party_button.png',
        'retire': 'retire_button.png',
        'new_game': 'new_game_button.png',
        'start_game': 'start_game_button.png',
        'start_game_2':'start_game_2.png',
        'start_game_3':'start_game_3.png'
         }
    
        for key, fn in term_templates.items():
            img = cv2.imread(os.path.join(templates_dir, fn), cv2.IMREAD_GRAYSCALE)
            if img is not None:
              self.skip_templates[key] = (img, img.shape[::-1], SCALE_GENERAL)

    def _detect_term_messages(self, gray):
        """تشخیص پیام‌های مرتبط با دوره‌های ریاست جمهوری"""
        # تشخیص پیام پایان دوره‌ها
        term_limit, conf = self._skip_match_with_confidence('term_limit', gray, 0.85)
        if term_limit:
            print("[TERM LIMIT] Presidential term limit reached")
            self.term_limit_reached = True
            return "term_limit"
        
        # تشخیص سایر پیام‌های مرتبط
        # ... می‌توان تمپلیت‌های بیشتری اضافه کرد ...
        
        return None
    
    def _get_window_screenshot(self):
        # region expects (left, top, width, height)
        region = (
            self.win.left,
            self.win.top,
            self.win.width,
            self.win.height
        )
        img = np.array(pyautogui.screenshot(region=region))
        return img
    

    def _start_new_game(self):
        """شروع ساده بازی جدید با ۳ کلیک روی تمپلیت‌ها"""
        
        # گرفتن اسکرین‌شات و پیدا کردن new_game
        gray = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
        btn1, conf1 = self._skip_match_with_confidence('new_game', gray, 0.8)
        print(f"[DEBUG] new_game: {btn1}, conf: {conf1:.4f}")
        if btn1:
            pyautogui.click(btn1)
            time.sleep(2)

        # گرفتن اسکرین‌شات و پیدا کردن start_game
        gray = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
        btn2, conf2 = self._skip_match_with_confidence('start_game', gray, 0.8)
        print(f"[DEBUG] start_game: {btn2}, conf: {conf2:.4f}")
        if btn2:
            pyautogui.click(btn2)
            pyautogui.click(btn2)
            pyautogui.moveTo(self._abs(REL_CORNER)); time.sleep(2)
            time.sleep(2)

        # گرفتن اسکرین‌شات و پیدا کردن start_game_2
        gray = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
        btn3, conf3 = self._skip_match_with_confidence('start_game_2', gray, 0.8)
        print(f"[DEBUG] start_game_2: {btn3}, conf: {conf3:.4f}")
        if btn3:
            pyautogui.click(btn3)
            pyautogui.click(btn3)
            pyautogui.moveTo(self._abs(REL_CORNER)); 
            time.sleep(2)
    
        gray = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
        btn3, conf3 = self._skip_match_with_confidence('start_game_2', gray, 0.8)
        print(f"[DEBUG] start_game_2: {btn3}, conf: {conf3:.4f}")
        if btn3:
            pyautogui.click(btn3)
            pyautogui.click(btn3)
            pyautogui.moveTo(self._abs(REL_CORNER)); 
            time.sleep(2)

        # st3, _ = self._skip_match_with_confidence('start_game_3', gray, SKIP_THRESHOLD)
        # if st3:
        #             print(f"[SKIP LOGIC] start_game_3 detected at {st3}, clicking once")
        #             pyautogui.click(st3)
        #             pyautogui.click(st3)
        #             print(f"[SKIP LOGIC] moving to REL_CORNER")
        #             pyautogui.moveTo(self._abs(REL_CORNER)); time.sleep(2)


        gray = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
        btn3, conf3 = self._skip_match_with_confidence('start_game_3', gray, 0.8)
        print(f"[DEBUG] start_game_3: {btn3}, conf: {conf3:.4f}")
        if btn3:
            pyautogui.click(btn3)
            pyautogui.click(btn3)
            pyautogui.moveTo(self._abs(REL_CORNER1))
            time.sleep(2)        

    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
            # اگر بازی تمام شده، شروع مجدد
        if self.game_over:
                print("[GAME] Starting new game...")
                self._start_new_game()
            
            # ریست‌کردن وضعیت
        self.term_count = 0
        self.term_limit_reached = False
        self.is_retired = False
        self.party_member = False
        self.game_over = False
        
        self.current_policies = 0
        self.target_policies = random.randint(2, 5)
        self.total_reward = 0 
        print(f"[RESET] Next turn target policies: {self.target_policies}")
        time.sleep(0.5)
        return self.take_screenshot_gray(), {}

    def take_screenshot_gray(self):
        img = pyautogui.screenshot()
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (84, 84))
        return small[np.newaxis, :, :].astype(np.uint8)
    
    def _compute_filled_ratio(self, bar_img):
            """
            نسبت پیکسل‌های رنگی (غیرسفید) به کل پیکسل‌های یک نوار را حساب می‌کند.
            bar_img: تصویر RGB یا BGR از یک نوار
            """
            # اگر BGR است به RGB تبدیل کن (یا برعکس، کافی است هر سه کانال را بررسی کنیم)
            img = bar_img.copy()
            # فرض می‌کنیم تصویر BGR است؛ اگر RGB بود این خط را بردارید:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 1) ماسک پیکسل‌های تقریبا سفید: 
            #    تعریف می‌کنیم پیکسل سفید = هر سه کانال بالای 240
            white_mask = np.all(img >= 230, axis=2)

            # 2) تعداد کل پیکسل‌ها
            total = img.shape[0] * img.shape[1]

            # 3) تعداد پیکسل‌های رنگی = کل - تعداد سفید
            colored = total - np.count_nonzero(white_mask)

            # 4) نسبت
            return colored / total
    
    def _extract_reward_from_screen(self):
        # … کد تشخیص reward box و ذخیره‌ی آن …
        # گرفتن اسکرین‌شات از پنجره بازی
        screen = self._get_window_screenshot()
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        # --- مرحله 1: پیدا کردن reward box ---
        # template matching روی تصویر خاکستری
        res = self._match_icon(gray, self.reward_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(f"[REWARD MATCH] confidence={max_val:.4f} center={max_loc}")
        if max_val < 0.8:
            print("[REWARD] reward box not found.")
            return 0.0

        # مختصات و ابعاد reward box
        x, y = max_loc
        h, w = self.reward_tpl.shape

        # --- مرحله 1.5: ذخیره‌ی تصویر reward box ---
        detected_box = screen[y:y + h, x:x + w]
        cv2.imwrite("detected_reward_box.png", detected_box)
        print("[REWARD] saved detected box to 'detected_reward_box.png'")

        # --------------- اینجا شروع اضافه کردن خواندن بارها ---------------

        # تعریف فهرست نام گروه‌ها به ترتیب نمایش
        groups = [
            "Popularity",
            "Capitalist", "Commuter", "Conservatives", "Environmentalist",
            "Ethnic Minorities", "Everyone", "Farmers", "Liberal",
            "Middle Income", "Motorist", "Parents", "Patriot",
            "Poor", "Religious", "Retired", "Self Employed",
            "Socialist", "State Employees"#, "Trade Unionist", "Wealthy",
            #"Youth"
        ]

        bar_values = {}

        # --- پردازش بارِ بزرگِ Popularity ---
        # مختصات و ابعاد ویژه‌ی Popularity
        pop_y0 =  113    # از بالای detected_box تا ابتدای نوار popularity
        pop_h  =  20    # ارتفاع بزرگ‌تر برای popularity
        pop_x0 =  60    # نقطه‌ی شروع افقی نوار popularity
        pop_x1 = 265    # نقطه‌ی پایان افقی کامل برای popularity

        pop_img = detected_box[pop_y0:pop_y0 + pop_h, pop_x0:pop_x1]
        filled_pop = self._compute_filled_ratio(pop_img)
        bar_values["Popularity"] = filled_pop
        print(f"[BAR] Popularity: {bar_values['Popularity']:.2f}")
        bar_values["Popularity"] = (filled_pop * 2) - 1.0
        #print(f"[DEBUG] Popularity shape={pop_img.shape}, filled_ratio_raw={filled_pop:.3f}, reward={(filled_pop * 2 - 1.0):.3f}")

        cv2.imwrite("bar_Popularity.png", pop_img)
        print(f"[BAR] Popularity: {bar_values['Popularity']:.2f}")
        #print(f"[BAR] Popularity filled: {filled_pop:.3f}, reward: {reward:.3f}")

        # --- پردازش سایر گروه‌ها با ابعاد یکنواخت ---
        start_y = pop_y0 + pop_h + 35  # اولین گروه دقیقاً زیر popularity + کمی فاصله
        bar_h   = 15
        gap_y   = 31
        x0 =  25
        x1 = 185

        for idx, name in enumerate(groups[1:]):  # از دومین گروه به بعد
            y0 = start_y + idx * gap_y
            y1 = y0 + bar_h
            bar_img = detected_box[y0:y1, x0:x1]

            bar_bgr = cv2.cvtColor(bar_img, cv2.COLOR_RGB2BGR)
            filled_ratio =self._compute_filled_ratio(bar_bgr)
            bar_values[name] = filled_ratio

            cv2.imwrite(f"bar_{name}.png", bar_img)
            print(f"[BAR] {name}: {filled_ratio:.2f}")

        # در نهایت بازگرداندن reward بر اساس Popularity
        return bar_values#["Popularity"]
    
    def _extract_budget_info(self):
        img = self._get_window_screenshot()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # برش نوار بالایی نسبت به پنجره
        bar_roi = gray[43:95, 12:419]
        _, thresh = cv2.threshold(bar_roi, 150, 255, cv2.THRESH_BINARY)
        cv2.imwrite("ocr_input_debug.png", thresh)

        text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')
        print("[OCR TEXT]\n", text)

        # --- به‌روز شده: ابتدا علامت منفی، بعد $ را می‌گیرد ---
        nums = re.findall(r'-?\$?\d{1,3}(?:,\d{3})*\.\d+', text)
        print("[DEBUG] Found numbers:", nums)
        if len(nums) < 3:
            print("[BUDGET INFO] Failed to find 3 numbers, got:", nums)
            return None, None, None

        def parse_bn(s):
            # s می‌تواند مثل '-$829.17' یا '$653.37' باشد
            negative = s.startswith('-')
            # حذف - و $ و فاصله و 'Bn'
            num = s.replace("-", "").replace("$", "").replace("Bn", "").replace(",", "").strip()
            val = float(num) * 1e9
            return -val if negative else val

        income      = parse_bn(nums[0])
        expenditure = parse_bn(nums[1])
        deficit     = parse_bn(nums[2])

        print(f"[BUDGET INFO] Income={income:.0f}, Expenditure={expenditure:.0f}, Deficit={deficit:.0f}")
        return income, expenditure, deficit

    # def _detect_election_result(self):
    #     """
    #     تشخیص نتیجه انتخابات با استفاده از OCR یا تطبیق الگو
    #     """
    #     try:
    #         # گرفتن اسکرین‌شات از ناحیه نتیجه انتخابات
    #         screen = self._get_window_screenshot()
    #         result_area = screen[209:1087, 396:542]  # تنظیم با توجه به رابط بازی
            
    #         # استفاده از OCR برای تشخیص متن
    #         text = pytesseract.image_to_string(result_area)
            
    #         if "won" in text.lower() or "re-elected" in text.lower():
    #             return "won"
    #         elif "lost" in text.lower() or "defeated" in text.lower():
    #             return "lost"
    #     except Exception as e:
    #         print(f"Error detecting election result: {str(e)}")
        
    #     return "unknown"

    def _fuzzy_match(self,text, target, threshold=0.7, label="(unnamed)"):
        ratio = difflib.SequenceMatcher(None, text.lower(), target.lower()).ratio()
        print(f"[DEBUG] Similarity with '{target}' in {label} region: {ratio*100:.1f}%")
        return ratio >= threshold



    def _detect_election_result(self):
        """تشخیص نتیجه انتخابات با OCR پیشرفته و منطق تطبیقی"""
        try:
            # 1. تنظیمات ثابت و مناطق تشخیص
            RESULT_REGION = (388, 208, 905, 261)  # x1, y1, x2, y2
            DEBUG_MODE = True  # فعالسازی ذخیره تصاویر برای دیباگ

            # 2. الگوهای تشخیص برد و باخت
            WIN_PATTERNS = [
                r"majority",  # غلط املایی رایج
                r"majorty",
                r"majority",
                r"votes? to remain",
                r"\d{1,3}% of votes",
                r"elected",
                r"win(s|ning)?"
            ]
            
            LOSE_PATTERNS = [
                r"voted out!?",
                r"votedout",
                r"eliminated",
                r"lose(s|ning)?",
                r"defeated",
                r"lost"
            ]

            # 3. گرفتن اسکرین‌شات و پیش‌پردازش
            screen = self._get_window_screenshot()
            gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            
            # 4. استخراج منطقه نتیجه
            x1, y1, x2, y2 = RESULT_REGION
            roi = gray[y1:y2, x1:x2]
            
            # 5. پیش‌پردازش پیشرفته
            # 5.1 افزایش کنتراست با CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            contrast_enhanced = clahe.apply(roi)
            
            # 5.2 حذف نویز
            denoised = cv2.fastNlMeansDenoising(contrast_enhanced, h=15)
            
            # 5.3 بزرگنمایی تصویر برای بهبود OCR
            scaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # 5.4 آستانه‌گیری تطبیقی
            processed = cv2.adaptiveThreshold(
                scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 8
            )
            
            # 6. اجرای OCR با تنظیمات بهینه
            custom_config = r'--psm 7 --oem 3 -l eng'
            text = pytesseract.image_to_string(
                processed, 
                config=custom_config,
                timeout=2
            ).strip()
            
            # 7. ذخیره تصاویر برای دیباگ
            if DEBUG_MODE:
                cv2.imwrite("election_roi_original.png", roi)
                cv2.imwrite("election_roi_processed.png", processed)
                print(f"[DEBUG] OCR Text: '{text}'")
            
            # 8. منطق تشخیص چند مرحله‌ای
            # 8.1 تشخیص سریع با تطابق دقیق
            for pattern in WIN_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    print(f"[DEBUG] Win detected by pattern: {pattern}")
                    return "won"
                    
            for pattern in LOSE_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    print(f"[DEBUG] Loss detected by pattern: {pattern}")
                    return "lost"
            
            # 8.2 تشخیص فازی با حساسیت بالا
            for pattern in WIN_PATTERNS:
                if self._fuzzy_match(text, pattern, 0.85):
                    print(f"[DEBUG] Win detected by fuzzy: {pattern}")
                    return "won"
                    
            for pattern in LOSE_PATTERNS:
                if self._fuzzy_match(text, pattern, 0.85):
                    print(f"[DEBUG] Loss detected by fuzzy: {pattern}")
                    return "lost"
            
            # 8.3 تشخیص بر اساس درصدها
            percentage_match = re.search(r'(\d{1,3})%', text)
            if percentage_match:
                percent = int(percentage_match.group(1))
                result = "won" if percent > 50 else "lost"
                print(f"[DEBUG] Percent detected: {percent}% -> {result}")
                return result
            
            # 9. استراتژی تکمیلی: تشخیص وجود متن
            if not text:
                print("[DEBUG] No text detected in region")
                return "unknown"
                
            # 10. استراتژی تکمیلی: آنالیز تراکم متن
            text_density = cv2.countNonZero(processed) / (processed.size / 255)
            if text_density > 0.3:  # منطقه حاوی متن است
                print(f"[DEBUG] Text density high ({text_density:.2f}) but no match")
                # تلاش برای تشخیص کلمات کلیدی در کاراکترهای جداگانه
                if any(char in text for char in ['W', 'w', 'M', 'm']):
                    return "won"
                elif any(char in text for char in ['L', 'l', 'V', 'v']):
                    return "lost"
            
            print("[DEBUG] No match found")
            return "unknown"

        except Exception as e:
            print(f"[ERROR] Election detection failed: {str(e)}")
            return "unknown"



    def _fuzzy_match(self, text, pattern, threshold, label=""):
        """انجام تطابق فازی با قابلیت دیباگ"""
        try:
            score = fuzz.partial_ratio(text.lower(), pattern.lower())
            if label:
                print(f"[FUZZY] {label}: '{text}' vs '{pattern}' = {score}")
            return score >= int(threshold * 100)
        except:
            return False
    

                
        # except Exception as e:
        #     print(f"[DEBUG] Error in election detection: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        
        # print("[DEBUG] Election result detection failed, returning unknown")
        # return "unknown"

    def _handle_term_limit(self):
        """مدیریت حالت پایان دوره‌های ریاست جمهوری"""
        print("[TERM LIMIT] Maximum terms reached. Making choice...")

        screen = np.array(pyautogui.screenshot())
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        retire_btn, retire_conf = self._skip_match_with_confidence('retire', gray, 0.8)
        if retire_btn and retire_conf > 0.8:
            pyautogui.click(retire_btn)
        # 1. تشخیص و کلیک روی گزینه‌ها
        choice = None
        start_time = time.time()
        
        while time.time() - start_time < 10:  # حداکثر 10 ثانیه انتظار
            screen = np.array(pyautogui.screenshot())
            gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            
            # تشخیص گزینه ادامه به عنوان عضو حزب
            party_btn, party_conf = self._skip_match_with_confidence('continue_as_party', gray, 0.8)
            if party_btn and party_conf > 0.8:
                pyautogui.click(party_btn)
                pyautogui.click(party_btn)
                choice = "continue_as_party"
                break
                
            # تشخیص گزینه بازنشستگی
            retire_btn, retire_conf = self._skip_match_with_confidence('retire', gray, 0.8)
            if retire_btn and retire_conf > 0.8:
                pyautogui.click(retire_btn)
                pyautogui.click(retire_btn)
                choice = "retire"
                break
                
            time.sleep(0.5)

        time.sleep(2)
        screen = np.array(pyautogui.screenshot())
        cv2.imwrite("debug_icon_c_screen.png", cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))

        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        icon_c_pos, icon_c_conf = self._skip_match_with_confidence('c', gray, 0.8)

        print(f"[MATCH] icon 'c' found: {bool(icon_c_pos)}, conf: {icon_c_conf:.2f}")

        if icon_c_pos and icon_c_conf > 0.8:
            #pyautogui.click(icon_c_pos)
            pyautogui.click(icon_c_pos)

        time.sleep(2)


        screen = np.array(pyautogui.screenshot())
        cv2.imwrite("debug_icon_c_screen.png", cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))

        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        icon_c_pos, icon_c_conf = self._skip_match_with_confidence('cont2', gray, 0.8)

        print(f"[MATCH] icon 'c' found: {bool(icon_c_pos)}, conf: {icon_c_conf:.2f}")

        if icon_c_pos and icon_c_conf > 0.8:
            #pyautogui.click(icon_c_pos)
            pyautogui.click(icon_c_pos)

        time.sleep(2)

        # 1. تشخیص و کلیک روی گزینه‌ها
        choice = None
        start_time = time.time()
        # 2. اگر گزینه‌ای تشخیص داده نشد، از منطق جایگزین استفاده کن
        if not choice:
            print("[TERM LIMIT] No option detected, using fallback logic")
            
            # بررسی موقعیت مکانی‌های شناخته شده برای گزینه‌ها
            known_positions = {
                "continue_as_party": (700, 500),
                "retire": (700, 600),
            }
            
            for option, pos in known_positions.items():
                # بررسی رنگ پیکسل‌های اطراف موقعیت
                pixel_color = screen[pos[1], pos[0]]
                if np.mean(pixel_color) > 200:  # اگر رنگ روشن باشد
                    pyautogui.click(pos)
                    choice = option
                    break
        
        return choice or "retire"  # پیش‌فرض بازنشستگی


    def step(self, action):

        if self.game_over or self.is_retired:
            return self.take_screenshot_gray(), 0, True, False, {}
        
       
        self.current_policies += 1
        
        # استخراج اکشن‌ها
        policy_idx   = int(action[0])
        slider_bin   = int(action[1])
        direction_float = -1.0 + 2.0 * slider_bin / (self.SLIDER_BINS - 1)

        name = self.policy_names[policy_idx]
        
        # 1. پاداش پایه برای اجرای هر سیاست
        base_reward = 0.0
        success_flags = {"policy_found": False, "slider_moved": False}
        
        # 2. ذخیره بودجه اولیه برای محاسبه تغییرات
        initial_budget = self._extract_budget_info()
        initial_deficit = initial_budget[2] if initial_budget else None

        # --- اجرای سیاست ---
        # کلیک روی آیکون policy
        pos = self._match_icon(self.policy_icons[name], tpl_name=name)
        if pos:
            pyautogui.click(pos)
            time.sleep(0.1)
            pyautogui.click(pos)  # دوبار کلیک برای اطمینان
            success_flags["policy_found"] = True
            base_reward += 0.1  # پاداش موفقیت‌آمیز پیدا کردن آیکون
        else:
            base_reward -= 0.2  # جریمه عدم پیدا کردن آیکون
            print(f"Failed to find policy icon: {name}")

        time.sleep(0.3)

        # حرکت slider به صورت پیوسته
        if success_flags["policy_found"]:
            try:
                # ۱) پیدا کردن slider knob
                print("[SLIDER] Trying to locate slider knob...")
                kn = self._match_icon(self.slider_tpl)
                if not kn:
                    print("[SLIDER][ERROR] Slider knob template not found!")
                else:
                    current_x, current_y = kn
                    print(f"[SLIDER] Knob found at x={current_x}, y={current_y}")
                    
                    # ۲) محاسبه گام حرکت
                    step = int(direction_float * 400)
                    target_x = max(SLIDER_MIN_X, min(SLIDER_MAX_X, current_x + step))
                    print(f"[SLIDER] direction={direction_float:.2f}, step={step}, target_x={target_x}")

                    # ۳) محدودیت‌های حرکت
                    # محاسبه‌ی گام و هدف اولیه
                    step = int(direction_float * 400)
                    raw_target_x = current_x + step

                    # کلَمپ کردن به بازه [SLIDER_MIN_X, SLIDER_MAX_X]
                    target_x = max(SLIDER_MIN_X, min(SLIDER_MAX_X, raw_target_x))

                    # درگ اسلایدر از current_x تا target_x
                    print(f"[SLIDER] Dragging knob from ({current_x},{current_y}) to ({target_x},{current_y})")
                    pyautogui.moveTo(current_x, current_y)
                    pyautogui.mouseDown()
                    pyautogui.moveTo(target_x, current_y, duration=0.3)
                    pyautogui.mouseUp()

                    # آپدیت وضعیت و جایزه
                    if target_x != current_x:
                        success_flags["slider_moved"] = True
                        base_reward += 0.2
                        print("[SLIDER] Move complete, reward +0.2")
                    else:
                        print(f"[SLIDER] Already at edge ({current_x}), no move needed")

            except Exception as e:
                print(f"[SLIDER][EXCEPTION] Error during slider move: {e}")

            time.sleep(0.2)

            # کلیک Apply و OK
            for tpl_name, tpl in [("apply", self.apply_tpl), ("ok", self.ok_tpl)]:
                try:
                    print(f"[BUTTON] Looking for '{tpl_name}' button...")
                    pos = self._match_icon(tpl)
                    if pos:
                        print(f"[BUTTON] '{tpl_name}' found at {pos}, clicking twice")
                        pyautogui.click(pos)
                        pyautogui.click(pos)
                        corner = self._abs(REL_CORNER1)
                        print(f"[BUTTON] Moving cursor away to {corner}")
                        pyautogui.moveTo(corner)
                        time.sleep(0.3)
                    else:
                        base_reward -= 0.1
                        print(f"[BUTTON][WARN] '{tpl_name}' button not found, reward -0.1")
                except Exception as e:
                    print(f"[BUTTON][EXCEPTION] Error handling '{tpl_name}' button: {e}")

        
        # 3. محاسبه تغییرات بودجه
        new_budget = self._extract_budget_info()
        new_deficit = new_budget[2] if new_budget else None
        deficit_change = 0.0
        
        if initial_deficit is not None and new_deficit is not None:
            deficit_change = initial_deficit - new_deficit  # مثبت یعنی کسری کاهش یافته
            scaled_change = np.clip(deficit_change / 5000.0, -1.0, 1.0)
            base_reward += scaled_change * 0.3

        # 4. پاداش مبتنی بر تعداد سیاست‌های اجرا شده
        progress_reward = min(1.0, self.current_policies / self.target_policies) * 0.2
        
        # 5. ترکیب پاداش‌های میانی
        intermediate_reward = base_reward + progress_reward
        
        # --- پردازش پایان نوبت ---
        if self.current_policies >= self.target_policies:
            print(f"[AUTO NEXT] Reached {self.target_policies}, skip logic")
            obs, popularity_data, done = self._perform_skip_logic()
            
            # 6. پاداش نهایی ترکیبی با استفاده از تمام داده‌ها
            final_reward = self._calculate_final_reward(
                popularity_data=popularity_data,
                intermediate=intermediate_reward,
                deficit_change=deficit_change
            )
            
            # ریست‌کردن برای نوبت جدید
            self.current_policies = 0
            self.target_policies = random.randint(2, 5)
            if done:
                self.game_over = True
            return obs, final_reward, done, False, {}
        
        return self.take_screenshot_gray(), intermediate_reward, False, False, {}

    def _calculate_final_reward(self, popularity_data, intermediate, deficit_change):
        """محاسبه پاداش نهایی با ترکیب عوامل مختلف"""
        if not popularity_data:
            print("[REWARD] No popularity data available!")
            popularity_component = 0.0
            group_component = 0.0
        else:
            # 1. مؤلفه محبوبیت رئیس‌جمهور (اصلی)
            president_popularity = popularity_data.get("Popularity", 0.0)
            popularity_component = (president_popularity * 2) - 1.0
            
            # 2. مؤلفه میانگین محبوبیت در گروه‌ها
            group_values = [
                val for name, val in popularity_data.items() 
                if name != "Popularity"
            ]
            if group_values:
                avg_group_popularity = sum(group_values) / len(group_values)
                group_component = (avg_group_popularity * 2) - 1.0
            else:
                group_component = 0.0
        
        # 3. مؤلفه تغییرات کسری بودجه
        deficit_component = np.clip(-deficit_change / 10000.0, -1.0, 1.0) if deficit_change else 0.0
        
        # وزن‌دهی به عوامل
        weights = {
            'president': 0.4,
            'groups': 0.3,
            'intermediate': 0.2,
            'deficit': 0.1
        }
        
        # ترکیب مولفه‌ها
        combined = (
            weights['president'] * popularity_component +
            weights['groups'] * group_component +
            weights['intermediate'] * intermediate +
            weights['deficit'] * deficit_component
        )
        
        # محدود‌کردن بازه پاداش
        final_reward = np.clip(combined, -1.0, 1.0)
        
        print(f"[FINAL REWARD] "
            f"President: {popularity_component:.2f}, "
            f"Groups: {group_component:.2f}, "
            f"Int: {intermediate:.2f}, "
            f"Def: {deficit_component:.2f} → "
            f"Total: {final_reward:.2f}")
        
        return final_reward
    
    def _perform_skip_logic(self):
            print("[SKIP LOGIC] Entering skip-turn")
            bar_values = {}  # مقدار پیش‌فرض
            done = False     # مقدار پیش‌فرض
            while True:
                gray = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)

                # popup
                pu, conf_pu = self._skip_match_with_confidence('popup', gray, POP_THRESHOLD)
                print(f"[SKIP LOGIC] popup confidence={conf_pu:.3f} at {pu}")
                if pu:
                    print(f"[SKIP LOGIC] popup detected at {pu}, conf={conf_pu:.3f}, clicking twice")
                    pyautogui.click(pu)
                    pyautogui.click(pu)
                    print(f"[SKIP LOGIC] moving to REL_CORNER")
                    pyautogui.moveTo(self._abs(REL_CORNER)); time.sleep(0.3)
                    continue

                # continue1
                c1, _ = self._skip_match_with_confidence('cont1', gray, SKIP_THRESHOLD)
                if c1:
                    print(f"[SKIP LOGIC] cont1 detected at {c1}, clicking once")
                    pyautogui.click(c1)
                    print(f"[SKIP LOGIC] moving to REL_CORNER")
                    pyautogui.moveTo(self._abs(REL_CORNER)); time.sleep(0.3)
                    continue

                #start_game_button
                # st3, _ = self._skip_match_with_confidence('start_game_3', gray, SKIP_THRESHOLD)
                # if st3:
                #     print(f"[SKIP LOGIC] start_game_3 detected at {c1}, clicking once")
                #     pyautogui.click(st3)
                #     pyautogui.click(st3)
                #     print(f"[SKIP LOGIC] moving to REL_CORNER")
                #     pyautogui.moveTo(self._abs(REL_CORNER)); time.sleep(0.3)
                #     continue
                # continue2 (بدون شرط قبلی)

                c2, _ = self._skip_match_with_confidence('cont2', gray, c2_THRESHOLD)
                if c2:
                    print(f"[SKIP LOGIC] cont2 detected at {c2}, clicking twice")
                    pyautogui.click(c2)
                    pyautogui.click(c2)
                    print(f"[SKIP LOGIC] moving to REL_CORNER1")
                    pyautogui.moveTo(self._abs(REL_CORNER1)); time.sleep(0.3)
                    # استخراج reward
                    bar_values = self._extract_reward_from_screen()
                    obs = self.take_screenshot_gray()
                    done = False  # بازی ادامه دارد
                    break

                # سایر آیکون‌ها
                ico = {}
                for k in ['e','f','f1','d','d1','d2','c','c1','a','b','g']:
                    if k == 'e':
                        thr = E_THRESHOLD  # مثلاً 0.88 یا مقدار دلخواه
                    elif k in ['d', 'd1', 'd2']:
                        thr = SKIP_DI_THRESHOLD
                    else:
                        thr = SKIP_THRESHOLD
                    pos, conf = self._skip_match_with_confidence(k, gray, thr)
                    ico[k] = pos
                    print(f"[SKIP LOGIC] icon {k} => {pos} (conf={conf:.3f})")
                
                # handle e
                if ico['e']:
                    print(f"[ELECTION] Election detected at {ico['e']}, processing...")
                    pyautogui.click(ico['e'])
                    pyautogui.click(ico['e'])
                    time.sleep(9)  # صبر برای لود شدن صفحه نتایج
                     # تشخیص نتیجه انتخابات
                    election_result = self._detect_election_result()
                    
                    # تشخیص پیام‌های دوره‌ها
                    gray = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
                    term_message = self._detect_term_messages(gray)
                    
                    if election_result == "won":
                        if term_message == "term_limit":
                            # پایان دوره‌های ریاست جمهوری
                            choice = self._handle_term_limit()
                            if choice == "retire":
                                print("[ELECTION] President retired. Game over.")
                                done = True
                            else:
                                print("[ELECTION] Continuing as party member.")
                                self.party_member = True
                                done = False
                        else:
                            choice = self._handle_term_limit()
                            # ادامه دوره ریاست جمهوری
                            print("[ELECTION] Continuing as president.")
                            done = False
                            
                    elif election_result == "lost":
                        print("[ELECTION] Lost the election. Game over.")
                        done = True
                    else:
                        print("[ELECTION] Unknown result. Assuming loss.")
                        done = True
                    
                    # استخراج نتایج و بازگشت
                    bar_values = self._extract_reward_from_screen()
                    obs = self.take_screenshot_gray()
                    return obs, bar_values, done
                
                # handel assasination
                if 'assassinated' in self.skip_templates:
                    pos_a, conf_a = self._skip_match_with_confidence('assassinated', gray, 0.9)
                    if pos_a:
                        print(f"[GAME OVER] Assassination detected at {pos_a} (conf={conf_a:.3f})")
                        obs = self.take_screenshot_gray()
                        try:
                            bar_values = self._extract_reward_from_screen()
                        except:
                            bar_values = {}
                        obs = self.take_screenshot_gray()
                        done = True  # بازی تمام شده است
                        break
                
                # پس از ساخت دیکشنری ico برای همه آیکون‌ها:
                corner_abs = self._abs(REL_CORNER)
                pos_g = ico.get('g')

                # اعتبارسنجی g تنها اگر دقیقاً روی REL_CORNER باشد
                valid_g = None
                if pos_g:
                    dx = abs(pos_g[0] - corner_abs[0])
                    dy = abs(pos_g[1] - corner_abs[1])
                    if dx <= 20 and dy <= 20:
                        valid_g = pos_g

                # منطق جدید برای g → c
                if valid_g:
                    while True:
                        print(f"[SKIP LOGIC] valid_g at {valid_g}, clicking REL_CORNER twice to dismiss g-popup")
                        pyautogui.click(corner_abs)
                        pyautogui.click(corner_abs)
                        time.sleep(0.5)

                        gray2 = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
                        pos_e, _ = self._skip_match_with_confidence('e', gray2, SKIP_THRESHOLD)
                        pos_c, _ = self._skip_match_with_confidence('c', gray2, SKIP_THRESHOLD)
                        pos_c1, _ = self._skip_match_with_confidence('c1', gray2, SKIP_THRESHOLD)

                        if pos_e:
                            print(f"[SKIP LOGIC] e detected at {pos_e} during g-loop, breaking to main loop")
                            break

                        if pos_c or pos_c1:
                            pos = pos_c or pos_c1
                            print(f"[SKIP LOGIC] c/c1 detected at {pos} during g-loop, clicking twice and closing box")
                            pyautogui.click(pos)
                            #pyautogui.click(pos)
                            pyautogui.click(corner_abs)
                            time.sleep(0.3)
                            break

                skip = False
                # handle f/f1
                if ico['f'] or ico['f1']:
                    pos_f = ico['f'] or ico['f1']
                    print(f"[SKIP LOGIC] f/f1 detected at {pos_f}, clicking twice")
                    pyautogui.click(pos_f)
                    pyautogui.click(pos_f)
                    print(f"[SKIP LOGIC] moving to REL_CORNER")
                    pyautogui.moveTo(self._abs(REL_CORNER))
                    skip = True

                # c+d priority
                elif (ico['c'] or ico['c1']) and (ico['d'] or ico['d1'] or ico['d2']):
                    pos_d = ico['d'] or ico['d1'] or ico['d2']
                    pos_c = ico['c'] or ico['c1']
                    print(f"[SKIP LOGIC] both c and d types detected, prioritizing d at {pos_d}, clicking twice")
                    pyautogui.click(pos_d)
                    pyautogui.click(pos_d)
                    pyautogui.click(pos_c)
                    pyautogui.click(pos_c)
                    print(f"[SKIP LOGIC] moving to REL_CORNER")
                    pyautogui.moveTo(self._abs(REL_CORNER))
                    skip = True

                # handle d
                elif ico['d']:
                    print(f"[SKIP LOGIC] d detected at {ico['d']}, clicking twice")
                    pyautogui.click(ico['d'])
                    pyautogui.click(ico['d'])
                    print(f"[SKIP LOGIC] moving to REL_CORNER")
                    pyautogui.moveTo(self._abs(REL_CORNER))
                    skip = True

                if not skip:
                    if ico['c'] or ico['c1']:
                        pos_c = ico['c'] or ico['c1']
                        print(f"[SKIP LOGIC] c/c1 detected at {pos_c}, clicking twice")
                        pyautogui.click(pos_c)
                        pyautogui.click(pos_c)
                    elif ico['a'] or ico['b']:
                        print(f"[SKIP LOGIC] a/b detected (none of target icons), clicking REL_CORNER")
                        pyautogui.click(self._abs(REL_CORNER))
                    else:
                        print(f"[SKIP LOGIC] no icon detected, clicking REL_CORNER")
                        pyautogui.click(self._abs(REL_CORNER))
                    print(f"[SKIP LOGIC] moving to REL_CORNER")
                    pyautogui.moveTo(self._abs(REL_CORNER)); time.sleep(0.3)

                # next-turn
                nt, _ = self._skip_match_with_confidence('next', gray, SKIP_THRESHOLD)
                if nt:
                    print(f"[SKIP LOGIC] next-turn detected at {nt}, clicking once")
                    pyautogui.click(nt)
                    print(f"[SKIP LOGIC] moving to REL_CORNER")
                    pyautogui.moveTo(self._abs(REL_CORNER)); time.sleep(0.3)
            bar_values = self._extract_reward_from_screen()
            obs = self.take_screenshot_gray()
            return obs, bar_values, done


    def _match_icon(self, tpl, tpl_name="tpl", multi_scale=True, scale_range=(0.8, 1.2), scale_step=0.1, debug=False):
        # گرفتن اسکرین‌شات و تبدیل به خاکستری
        screenshot = np.array(pyautogui.screenshot())
        gray = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)
        
        best_confidence = -1
        best_center = None
        best_scale = 1.0
        best_w, best_h = tpl.shape[::-1]
        best_max_loc = (0, 0)
        
        # محدوده‌های مقیاس برای تطبیق چندمقیاسی
        scales = []
        current_scale = scale_range[0]
        while current_scale <= scale_range[1]:
            scales.append(current_scale)
            current_scale += scale_step
            current_scale = round(current_scale, 2)  # جلوگیری از خطاهای ممیز شناور

        for scale in scales:
            # محاسبه ابعاد جدید بر اساس مقیاس
            new_w = int(tpl.shape[1] * scale)
            new_h = int(tpl.shape[0] * scale)
            
            # بررسی ابعاد معتبر
            if new_w < 5 or new_h < 5 or new_w > gray.shape[1] or new_h > gray.shape[0]:
                continue
                
            try:
                # تغییر اندازه تمپلیت با درون‌یابی کیفیت بالا
                resized_tpl = cv2.resize(
                    tpl, 
                    (new_w, new_h), 
                    interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                )
                
                # تطبیق تمپلیت با روش‌های مختلف
                res = cv2.matchTemplate(gray, resized_tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                # به‌روزرسانی بهترین نتیجه
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_scale = scale
                    best_w, best_h = new_w, new_h
                    best_max_loc = max_loc
                    best_center = (max_loc[0] + new_w//2, max_loc[1] + new_h//2)
                    
            except Exception as e:
                print(f"[خطا در مقیاس {scale:.2f}]: {str(e)}")
                continue

        # اعتبارسنجی نهایی و دیباگ
        print(f"[ICON MATCH] {tpl_name}: confidence={best_confidence:.4f} scale={best_scale} center={best_center}")
        #time.sleep(1)
        if debug and best_confidence > 0.3:  # ذخیره تصویر فقط اگر نتیجه معقول باشد
            debug_img = screenshot.copy()
            top_left = best_max_loc
            bottom_right = (top_left[0] + best_w, top_left[1] + best_h)
            cv2.rectangle(debug_img, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                debug_img, 
                f"{tpl_name}: {best_confidence:.2f}", 
                (top_left[0], top_left[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
            cv2.imwrite(f"debug_{tpl_name}.png", debug_img)

        return best_center if best_confidence >= self.CONFIDENCE else None


    def _skip_match_with_confidence(self, key, gray, thr):
        if key not in self.skip_templates:
            return None, 0.0
        tpl, (w, h), sc = self.skip_templates[key]
        small = cv2.resize(gray, (0,0), fx=sc, fy=sc)
        res = cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv >= thr:
            return (int((ml[0] + w//2) / sc), int((ml[1] + h//2) / sc)), mv
        return None, mv

    def _abs(self, rel):
        return (self.win.left + rel[0], self.win.top + rel[1])

    def render(self):
        pass

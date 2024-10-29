import pyautogui
import pytesseract
import random
import time
import re
import PIL
import os
import cv2
import numpy as np
import dlib
import pyscreenshot as ImageGrab
import pynput
import string
from datetime import datetime
import os.path
import easyocr
from PIL import Image, ImageOps, ImageFilter
import csv

from string import punctuation
import time

'''
Датасайнтист, гитарист и любитель лис - это я. Увлечен AI, классическим искусством, турничком и готовкой здоровых вкусняшек.

Люблю исследовать серьезные вещи и громко ржать. Тайная страсть - писать короткие фентези в стиле документалистики, иногда на английском тк второй родной.

Классно, когда женщина самостоятельна, образованна, в меру активна и романтична. Давай будем чувствительны друг к другу и трезвы к реальности.

Допустим пошли на прогулку в лес, допиши сюжет и детали не подглядывая.
'''


'''
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
# Download the stopwords if not already downloaded
nltk.download('stopwords')
# Get the Russian stop words from NLTK
stop_words = set(stopwords.words('russian'))
def get_negativity_score(text):
    sid = SentimentIntensityAnalyzer()
    cleaned_text = ' '.join([word for word in text.split() if word.lower() not in stop_words and word not in punctuation])
    scores = sid.polarity_scores(cleaned_text)
    negativity_score = (scores['neg'] * 10)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("NEG SCORE:",negativity_score)
    print(text)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    return round(negativity_score, 1)
'''


#######################

# reresh icon
refx, refy = 814, 110

# level 1 unmatch
nox1, noy1 = 1397, 820

# info icon
infox, infoy = 1633, 725

# level 2 unmatch
nox2, noy2 = 1405, 823

# lelel 2 yes
yesx2, yesy2 =  1559, 820

# full rectangle
x1, y1, x2, y2 = 1309, 208, 1675, 787

# text area
tx1, ty1, tx2, ty2 = 1307, 505, 1676, 775

##########################

# полная картинка профиля без нижних кнопок (верхний левый угол, нижний правый угол)
# full rectangle
x1, y1, x2, y2 = x1, y1, x2, y2


# face detector
fx1, fy1, fx2, fy2 = 1289, 210, 1653, 791

# green image area
x1, y1, x2, y2 = 1289, 210, 1653, 791





#con





import pyautogui
import random
import string


import pyautogui
import time
import keyboard

import pyperclip
import time
import pyautogui

import cv2
import numpy as np
import pytesseract
import pyautogui
import os
import random
import string

def time_left():
    # Define the coordinates of the screen rectangle
    x1, y1, x2, y2 = tx1, ty1, tx2, ty2

    # Convert the time pattern to an image using the specified font and size
    time_pattern = "hh:mm:ss"
    pattern_img = np.zeros((100, 100, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_size = cv2.getTextSize(time_pattern, font, font_scale, 2)[0]
    text_x = int((pattern_img.shape[1] - text_size[0]) / 2)
    text_y = int((pattern_img.shape[0] + text_size[1]) / 2)
    cv2.putText(pattern_img, time_pattern, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    # Take a screenshot of the whole screen
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Convert the pattern image to grayscale
    pattern_gray = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY)

    # Find the pattern in the screenshot
    threshold = 0.6
    res = cv2.matchTemplate(gray, pattern_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # Iterate through the locations and extract the time strings
    time_strings = []
    for pt in zip(*loc[::-1]):
        x, y = pt[0], pt[1]
        time_img = gray[y:y+pattern_gray.shape[0], x:x+pattern_gray.shape[1]]
        _, time_thresh = cv2.threshold(time_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        time_string = pytesseract.image_to_string(time_thresh, config='--psm 11')
        time_strings.append(time_string)
        print(time_string)

    # Return the time string(s)
    return time_strings

def copy_info_text():
    # Move the mouse to the specified position and click

    #infox, infoy =1655, 726 #find_icon_on_screen("info_icon.png")
    pyautogui.moveTo(infox, infoy) # find_icon_on_screen("info_icon.png") )
    pyautogui.click()
    time.sleep( 3 )

    # Wait for the context menu to appear and select "Select All"
    time.sleep(0.5)
    pyperclip.copy('')
    pyautogui.hotkey('ctrl', 'a')

    # Wait for the text to be selected and copy it to the clipboard
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'c')

    # Wait for the copy operation to complete and retrieve the contents of the clipboard
    time.sleep(0.5)
    s = pyperclip.paste()
    allowed_chars = r"[a-zA-Zа-яА-Я.,!()? ]"
    cleaned_text = re.sub(f"[^{allowed_chars}]", "", s)
    #print(s)
    return cleaned_text

def save_screenshot(pr):


    # Capture a screenshot of the screen rectangle
    im = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))

    # Generate a random string to use in the file name
    random_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    # Save the screenshot to a file with the random string in the name
    file_name = f"{pr}_{random_text}.png"
    im.save(file_name)

    return file_name



def detect_female_face_on_screenshot():
    x1, y1, x2, y2 = fx1, fy1, fx2, fy2

    # Capture a screenshot of the screen rectangle
    img = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    img = np.array(img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize the image to reduce the impact of illumination and contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Load the face and eye classifiers
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Check if a clear female face is present with eyes
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) < 2:
            continue

        # Save the image with the detected face and eyes
        filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        cv2.imwrite(f"faces/GOOD_{filename}.jpg", roi_color)
        print("FACE DETECTED")
        return True

    # Save the screenshot to a file if no clear female face with eyes is detected
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    cv2.imwrite(f"faces/BAD_noface_{filename}.jpg", img)

    print("NO FACE")
    return False

'''
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2
import numpy as np

def detect_female_face_on_screenshot_EMOTION():
    x1, y1, x2, y2 = fx1, fy1, fx2, fy2

    # Capture a screenshot of the screen rectangle
    img = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    img = np.array(img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the face detector
    detector = MTCNN()

    # Detect faces
    faces = detector.detect_faces(img)

    # Load the VGGFace model and define the gender classes
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    gender_classes = ['male', 'female']

    # Check if a clear female face is present with eyes and no negative facial emotion
    for face in faces:
        x, y, w, h = face['box']
        keypoints = face['keypoints']

        # Check if the face is female
        face_img = img[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = preprocess_input(face_img, version=2)
        gender_probs = model.predict(face_img)
        gender = gender_classes[np.argmax(gender_probs)]
        if gender != 'female':
            continue

        # Check if the facial emotion is negative
        emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg', classes=len(emotion_classes))
        emotion_probs = emotion_model.predict(face_img)
        emotion = emotion_classes[np.argmax(emotion_probs)]
        if emotion in ['sad', 'angry', 'surprise']:
            continue

        # Save the image with the detected face
        filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        cv2.imwrite(f"faces/GOOD_{filename}.jpg", img)
        return True

    # Save the screenshot to a file if no clear female face with eyes is detected
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    cv2.imwrite(f"faces/BAD_noface_{filename}.jpg", img)
    return False

'''

def check_screenshot_for_color():

    x1, y1, x2, y2 = ox1, oy1, ox2, oy2

    # Capture a screenshot of the screen rectangle
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))

    #screenshot = pyautogui.screenshot()
    pixel_color = (124, 253, 163) # the color #7cfda3 in RGB format
    pixel_count = 0

    # loop through the pixels in the screenshot and count the number of pixels that match the desired color
    for x in range(screenshot.width):
        for y in range(screenshot.height):
            if screenshot.getpixel((x, y)) == pixel_color:
                pixel_count += 1
                if pixel_count == 9:  # if we find 9 consecutive pixels with the desired color, we can exit early and return True
                    return True

    # if we didn't find the desired pattern of pixels, return False
    #print("no green color found")
    return False



def preprocess_image(image_path):

    # Create the EasyOCR reader
    reader = easyocr.Reader(['en'])
    # Get the text from the cleaned image
    eng = reader.readtext(image_path)
    active = False
    for item in eng:
        if  "online" in item[1].lower(): # or "active" in item[1].lower(): #recently active" or "online now"
            #print("Found 'Recently Active'")
            active = True
            break

    # for Russian need to preprocess_image
    ##########

    with Image.open(image_path) as image:
        # Convert the image to black and white
        bw_image = image.convert('L')

        # Invert the colors
        inverted_image = ImageOps.invert(bw_image)

        # Binarize the image
        threshold_value = 100
        binarized_image = inverted_image.point(lambda x: 0 if x < threshold_value else 255)

        # Save the image
        binarized_image.save(image_path)
        rus = pytesseract.image_to_string(Image.open(image_path), lang='rus')



    return eng, rus, image_path, active




def check_text():
    # Define the coordinates of the screen rectangle
    x1, y1, x2, y2 = tx1, ty1, tx2, ty2

    # Capture a screenshot of the screen rectangle
    im = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
    txt_file  = os.path.join('faces', "txt_" + ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=8))) + '.png'
    im.save(txt_file)
    eng,rus, filename, active = preprocess_image(txt_file)


    filename_txt = 'desc.txt' # _' + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + '.txt'
    with open(os.path.join('faces', filename_txt), 'a', encoding='utf-8') as f:
        f.write(rus)

    return eng, rus, filename, active



def find_icon_on_screen(icon_path, threshold=0.6):
    # Load the icon image
    icon = cv2.imread(icon_path)
    icon_gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)

    # Take a screenshot of the whole screen
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Find the icon in the screenshot
    res = cv2.matchTemplate(screenshot_gray, icon_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    if len(loc[0]) == 0:
        return None

    # Calculate the center of the icon
    icon_w = icon.shape[1]
    icon_h = icon.shape[0]
    center_x = int(loc[1][0] + icon_w/2)
    center_y = int(loc[0][0] + icon_h/2)


    return center_x, center_y


import cv2
import numpy as np
import pytesseract
import pyautogui



def find_icon_on_screen_bw(icon_path, threshold=0.6):
    # Load the icon image
    icon = cv2.imread(icon_path)
    icon_gray = cv2.cvtColor(icon, cv2.COLOR_BGR2GRAY)
    _, icon_bw = cv2.threshold(icon_gray, 0, 255, cv2.THRESH_BINARY)

    # Take a screenshot of the whole screen
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    _, screenshot_bw = cv2.threshold(screenshot_gray, 0, 255, cv2.THRESH_BINARY)

    # Find the icon in the screenshot
    res = cv2.matchTemplate(screenshot_bw, icon_bw, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    if len(loc[0]) == 0:
        return None

    # Calculate the center of the icon
    icon_w = icon.shape[1]
    icon_h = icon.shape[0]
    center_x = int(loc[1][0] + icon_w/2)
    center_y = int(loc[0][0] + icon_h/2)

    find_time_on_screen()

    return center_x, center_y

#print( find_icon_on_screen("info_icon.png") )
#exit()

def log_description(s):
    filename_txt = 'desc.txt' # _' + ''.join(random.choices(string.ascii_letters + string.digits, k=8)) + '.txt'
    with open(filename_txt, 'a', encoding='utf-8') as f:
        f.write('\n-----------------' + " SIZE " + str(len(s)) +"\n" )
        f.write(s)
        f.write('\n-----------------------------------------------\n')
    f.close()



def no_level1():
    #find_icon_on_screen("level1_no.png")
    pyautogui.click(nox1,noy1)

def no_level2():
    x, y = 975,984 #find_icon_on_screen("level2_no.png")
    pyautogui.click(nox2, noy2)

def ok_level2():

    #find_icon_on_screen("level2_yes.png")
    #x, y = find_icon_on_screen("level2_no.png") # fo testing purposes
    pyautogui.click(yesx2, yesy2 )


def refresh_browser():

    #find_icon_on_screen("refresh.png")
    pyautogui.click(refx, refy)

def good_woman_save():
    import random
    import string
    import pyautogui

    # Generate random file name
    file_name = 'GOOD_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.png'

    # Take screenshot of whole screen
    screenshot = pyautogui.screenshot()

    # Save screenshot with random file name
    screenshot.save("./good/"+file_name)


def check_goodwords(s):
    root_positive_keywords = [
    "увлеч", "активн", "образован", "искрен", "добр", "честн", "юмор",
    "оптимист", "творч", "любознател", "путешеств", "общител", "сем",
    "вегетариан", "эколог", "волонт", "спорт", "заботлив", "целеустремле", "целеустремлё",
    "открыт", "энергичн", "уважитель", "музык", "приключен", "книг", "позитивн",
    "природ", "готов", "кино", "театр", "дружелюб", "любопытств",
    "фитнес", "разнообразн", "благодар", "вдохновля", "духовн", "эмпат", "жизнерадост",
    "здоров", "уравновеш", "искател", "коммуникабел", "компромисс", "креатив",
    "легк", "любящ", "мечтател", "мотивац", "находчив", "независим", "отзывч",
    "позитивн", "решител", "смел", "смышлен", "сочувств", "страст", "талантлив",
    "умн", "уникал", "харизматичн", "чувствител", "шут", "экстраверт", "ярк"]

    for sw in root_positive_keywords:
        if sw in s:
            return(sw)

    return None

def check_stopword(s):
    stop = ["инстаграм", "instagram", "инст:", "inst", "insta", "@", "//t.me", "http", "www", "рекрут", "recruit", "заботлив", "инста", "поступки", "развод", "щедр", "адекват", "психолог", "влюбиться", "не люблю", "гештальт", "развитие", "развиваться", "саморазвитие", "душн", "эзотерик", "дочь", "дочень", "ребенок", "сын","сынок", "зануд", "ребёнок", "риэлтор", "консультант", "коуч", "первая не пишу", "тут редко", "серьёзные отношения","цветы", "цветов", "руководител", "сразу в бан","!!!","!!", "ч/ю","озабоченн","настоящая женщина", "настоящий мужчина", "вредная", "капризная","научи","приоритет","психотерапевт","психоанали","хам","мужик", "как дела", "настойчивость", "не отвечаю", "не пишу перв", "начальн", "не пишу", "не пишу первой",
    "тут бываю редко","госпожа","ответственн","дочк", "свидани", "пиши первый", "мужчина должен", "настоящий мужчина", "на хрен", "идите", "выношу мозг","модель", "приветствуется", "высок", "мужчины", "злая", "вредн","ухажива", "здесь редко", "извращ", "есть тут", "сериал", "жадны", "ценю", "ig:", "тут редко", "мразь", "интересует общение", "отца не ищу", "папу не ищу", "папу им не ищу",
    "отца им не ищу", "псих", "одноразовы", "собак", "определяет цель", "влюбиться", "ничего решать", "не ищу", "мама", "невыносим", "конкретн", "предложен", "просьба",
    "встреча определяет", "свидан", "прошу", "характер", "девушк", "общени", "переписки", "характер", "ведьм", "прицеп", "выше", "достойн", "мама", "самого лучшего", "скучн","скуча","соскуч", "знакомств", "напиши", "просьба не", "если ты", "рост"]

    for sw in stop:
        if sw in s:
            return(sw)

    return None

import cv2
import pyautogui




def rand_ref():
    num = random.randint(1, 999)
    # Check if the number is divisible by N
    if num % 3 == 0:
        refresh_browser()
        time.sleep( 7 )


def wait_before_continue(h,m):

    import time
    print("Waiting to start at",f"{h}:{m}")
    while True:
        # Get the current time
        current_time = time.localtime()
        if current_time.tm_hour == h and current_time.tm_min >= m:
            print("Starting",current_time)
            break

        # Wait for one minute before checking again
        time.sleep(60)


def check_maxlikes():
    if find_icon_on_screen("tinder_plus.png") != None:
        print("Ура! На сегодня усё!")
        os.system('systemctl suspend')
        time.sleep( 7 )
        return True
    else:
        return False


def log_stats():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    day_of_week_str = now.strftime("%A")
    # TODO: fix all these formulas below and make sure it saves everything!
    elapsed_time = time.time() - start_time
    avg_time_per_profile = elapsed_time / good_profiles

    with open('stats.csv', mode='a') as stats_file:
        stats_writer = csv.writer(stats_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        stats_writer.writerow([current_time, day_of_week_str, good_profiles, bad_profiles, round(good_profiles/bad_profiles,2), \
            round(100*good_profiles/(good_profiles+bad_profiles),2), avg_time_per_profile, 100 * avg_time_per_profile / 60 / 60, elapsed_time/60 ])

    print("~ STAT ~ GOOD/BAD profiles", good_profiles, "/", bad_profiles, "| ratio =", round(good_profiles/bad_profiles,2), "| Годные =", round(100*good_profiles/(good_profiles+bad_profiles),2),  avg_time_per_profile / 60,   )

    print("Time spent (in minutes):", elapsed_time / 60)
    print("Average time per profile (in minutes):", avg_time_per_profile / 60)
    print("To get 100 good profiles (in hours):", 100 * avg_time_per_profile / 60 / 60)

import plotext as plt
keyword_stat = {}

def update_dict_value(my_dict, my_key):
    if my_key in my_dict:
        my_dict[my_key] += 1
    else:
        my_dict[my_key] = 1
    return my_dict



def plot_horizontal_barchart(data, title=""):
    categories, values = zip(*data.items())

    plt.bar(categories, values, orientation="horizontal") # width=3 / 5
    plt.title(title)
    plt.show()

'''
from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
scheduler.add_job(log_stats, 'cron', minute='0,5,10,15,20,25,30,35,40,45,50,55')
scheduler.start()
'''

min_text_len = 150 # rus character threshold
bad_profiles = 0
good_profiles = 0
start_time = time.time()

waiting_sessions = 0

#nltk.download('vader_lexicon')
prev_description = ""

import sys
waited = False


stat_offline = 0
stat_active = 0
stat_wrong_age = 0
stat_good_profile = 0
stat_stopword = 0
stat_short_desc = 0
stat_bad_name = 0
stat_short_name = 0
stat_too_far = 0



while (True):
    #if check_maxlikes(): break

    if not waited:
        if len(sys.argv) > 1:
            tt = sys.argv[1].split(":")
            wait_before_continue(int(tt[0]),int(tt[1]))
    waited = True

    try:
        now = datetime.now()
        print('// Current DateTime:', now)

        dict_stats = {

                        "GOOD PROFILE":stat_good_profile,
                        "active":stat_active,
                        "offline":stat_offline,
                        "wrong age":stat_wrong_age,
                        "stopword":stat_stopword,
                        "short descr":stat_short_desc,
                        "bad name":stat_bad_name,
                        "short name":stat_short_name,
                        "too far":stat_too_far,
                    }

        #plot_horizontal_barchart( keyword_stat )
        print(dict_stats)
        print(keyword_stat)
        #plot_horizontal_barchart( dict_stats )

        while find_icon_on_screen("me.png"):
            if waiting_sessions < 3:
                tt = random.randint(1000, 5000)/100
            else:
                print("waiting loger now..")
                tt = random.randint(5000, 8000)/100  # wait longer after 2 tries

            print("- no profiles, WAITING", tt)
            time.sleep( tt )
            refresh_browser()
            time.sleep( 8 )
            waiting_sessions = waiting_sessions+1

        waiting_sessions = 0 # reset the counter if we got here

        if bad_profiles!=0 and good_profiles > 0:
            print("~ STAT ~ GOOD/BAD profiles", good_profiles, "/", bad_profiles, "| ratio =", round(good_profiles/bad_profiles,2), "| Годные =", \
                                                                                                        round(100*good_profiles/(good_profiles+bad_profiles),2) )
            elapsed_time = time.time() - start_time
            avg_time_per_profile = elapsed_time / good_profiles
            print("Time spent (in minutes):", elapsed_time / 60)
            print("Average time per profile (in minutes):", avg_time_per_profile / 60)
            print("To get 100 good profiles (in hours):", 100 * avg_time_per_profile / 60 / 60)

        # just for testing  purposes we turn it off
        #green_color = find_icon_on_screen("active.png") or active # gree light

        '''
        eng, rus, filename, active = check_text()
        if not(active):
            print("- в оффлайне")
            no_level1()
            bad_profiles += 1
            stat_offline += 1
            #rand_ref()
            continue
        else:
            print("[+] активна")
            stat_active += 1
        '''


        s = copy_info_text()
        s = s.lower()
        if prev_description==s:
            refresh_browser()
        else:
            prev_description=s

        name_min_len = 3
        imya = s.split("\n")[0]

        max_age = 39
        age = int(s.split("\n")[1])
        if age > max_age:
            print(f"- возраст больше {max_age}", imya, age)
            no_level2()
            time.sleep( 2 )
            bad_profiles += 1
            #rand_ref()
            stat_wrong_age += 1
            continue

        #print("- name",imya)
        eng_rus_only = r'^[A-Za-zА-Яа-яЁё\s]+$' # rus/eng, can include spaces
        name_stop = ["miss","имя"]

        if not bool(re.match(eng_rus_only, imya)) or any(word in imya for word in name_stop):
            print("- хреновое имя", imya)
            no_level2()
            time.sleep( 2 )
            bad_profiles += 1
            stat_bad_name += 1
            #rand_ref()
            continue
        elif (len(imya)<name_min_len):
            print("- короткое имя", imya)
            no_level2()
            time.sleep( 2 )
            bad_profiles += 1
            stat_short_name += 1
            #rand_ref()
            continue

        if len(s) < min_text_len:
            print("- слишком короткое описание",s)
            no_level2()
            time.sleep( 2 )
            bad_profiles += 1
            stat_short_desc += 1
            #rand_ref()
            continue

        match = re.search(r'\b(\d{1,3}) kilometers away\b', s, re.MULTILINE)
        num = int(match.group(1))
        dist = 70

        if (match and num <= dist) or "moscow" in s or "москва" in s:
            print("[+] расстояние норм или Москва/Moscow:",num)
        else:
            print("- далековато...")
            no_level2()
            time.sleep( 2 )
            bad_profiles += 1
            #rand_ref()
            stat_too_far += 1
            continue


        sw = check_stopword(s)
        if sw != None:
            print("- СТОП-СЛОВО detected!",sw)
            no_level2()
            time.sleep( 2 )
            bad_profiles += 1
            stat_stopword += 1

            keyword_stat = update_dict_value(keyword_stat, sw)


            continue



        '''
        sw = check_goodwords(s)
        if sw != None:
            print("[+] ГОДНОСТИ СЛОВО detected!",sw)
        else:
            print("- Нету ГОДНОСТИ-СЛОВ!",sw)
            no_level2()
            time.sleep( 2 )
            bad_profiles += 1
            #rand_ref()
            continue
        '''



        #get_negativity_score(s)
        #no_level2()
        #continue




        now = datetime.now()
        print('[+] ACTIVE & FACE & имя и описание норм', now.strftime("%Y-%m-%d %H:%M:%S") )
        print(s)
        print("--------------------------------------------------")
        log_description( s )
        time.sleep( 2 )
        #good_woman_save()
        x, y = find_icon_on_screen("level2_yes.png")
        #x, y = find_icon_on_screen("level2_no.png") # for testing purposes
        pyautogui.click(x,y)
        time.sleep( 3 )
        stat_good_profile += 1
        good_profiles += 1

        # TODO: plot stats here
        #plot_horizontal_barchart(dict_stats)
        #plot_horizontal_barchart(keyword_stat)
        if check_maxlikes(): break
        #print(dict_stats)
        #print(keyword_stat)

    except:

        refresh_browser()
        if check_maxlikes(): break

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

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from string import punctuation

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


# full rectangle
x1, y1, x2, y2 = 841, 167, 1548, 960

# face area
fx1, fy1, fx2, fy2 = 841, 171, 1405, 766

# green image area
ox1, oy1, ox2, oy2 = 837, 592, 901, 876

# text area
tx1, ty1, tx2, ty2 = 836, 629, 1497, 878

# yes/no locations
yesx, yesy = 1339, 972
nox, noy = 1050, 979

# level 0 unmatch
nox0, noy0 = 1057, 924

# info icon
infox, infoy = 1388, 836

import pyautogui
import random
import string


import pyautogui
import time
import keyboard

import pyperclip
import time
import pyautogui

def copy_info_text():
    # Move the mouse to the specified position and click

    infox, infoy = find_icon_on_screen("info_icon.png")
    pyautogui.moveTo(infox, infoy) # find_icon_on_screen("info_icon.png") )
    pyautogui.click()

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
        #print("FACE DETECTED")
        return True

    # Save the screenshot to a file if no clear female face with eyes is detected
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    cv2.imwrite(f"faces/BAD_noface_{filename}.jpg", img)

    #print("NO FACE")
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
        if "Recently Active" in item[1]:
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
    x, y = find_icon_on_screen("level1_no.png")
    pyautogui.click(x,y)

def no_level2():
    x, y = find_icon_on_screen("level2_no.png")
    pyautogui.click(x,y)

def ok_level2():
    x, y = find_icon_on_screen("level2_yes.png")
    #x, y = find_icon_on_screen("level2_no.png") # fo testing purposes
    pyautogui.click(x,y)


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


def check_stopword(s):
    stop = ["инстаграм", "instagram", "инст:", "inst", "insta", \
            "@", "//t.me", "http", "www", "рекрут", "recruit", "заботлив", "инста", "поступки", "развод", "щедр", "адекват", "психолог", "влюбиться", \
            "не люблю", "гештальт", "развитие", "развиваться", "саморазвитие", "душн", "эзотерик", "дочь", "дочень", "ребенок", "сын","сынок",\
            "зануд", "ребёнок", "риэлтор", "консультант", "коуч", "первая не пишу", "тут редко", "серьёзные отношения","цветы", "цветов",\
                "руководител","сразу в бан","!!!","!!", "ч/ю","озабоченн","настоящая женщина", "настоящий мужчина", "вредная", "капризная",\
            "научи","приоритет","психотерапевт","психоанали","хам","мужик", "как дела", "настойчивость", "не отвечаю", "не пишу", "не пишу первой", "госпожа","ответственн","дочк", "свидани", "пиши первый", "мужчина должен", "настоящий мужчина", "на хрен", "идите", "выношу мозг","модель"]

    for sw in stop:
        if sw in s:
            return(sw)

    return None



min_text_len = 150 # rus character threshold
bad_profiles = 0
good_profiles = 0

nltk.download('vader_lexicon')
while (True):


    #if bad_profiles!=0:
    #    print("~ STAT ~ GOOD/BAD profiles", good_profiles, "/", bad_profiles, " | ratio = ", good_profiles/bad_profiles)

    while find_icon_on_screen("me.png"):
        tt = random.randint(1000, 5000)/100
        print("- no profiles, WAITING", tt)
        time.sleep( tt )
        x, y = 927, 96 #refresh
        pyautogui.click(x,y)
        time.sleep( 7 )

    # just for texting purposes we turn it off\
    #eng, rus, filename, active = check_text()
    #green_color = find_icon_on_screen("active.png") or active # gree light
    '''
    if not(active):
        print("- в оффлайне")
        no_level1()
        bad_profiles += 1
        continue
    else:
        print("[+] активна")
    '''

    '''
    fem = detect_female_face_on_screenshot()
    if not(fem):
        print("- нет женского лица на картинке")
        no_level1()
        bad_profiles += 1
        continue
    '''
    s = copy_info_text()
    s = s.lower()


    sw = check_stopword(s)
    if sw != None:
        #print("- СТОП-СЛОВО detected!",sw)
        no_level2()
        bad_profiles += 1
        continue

    if len(s) < min_text_len:
        #print("- слишком короткое описание",s)
        no_level2()
        bad_profiles += 1
        continue

    get_negativity_score(s)
    no_level2()
    continue

    name_min_len = 3
    imya = s.split("\n")[0]
    #print("- name",imya)
    eng_rus_only = r'^[a-zA-Zа-яА-Я]+$'
    name_stop = ["miss","имя"]

    if not bool(re.match(eng_rus_only, imya)) or any(word in imya for word in name_stop):
        print("- хреновое имя", imya)
        no_level2()
        bad_profiles += 1
        continue
    elif (len(imya)<name_min_len):
        print("- короткое имя", imya)
        no_level2()
        bad_profiles += 1
        continue


    now = datetime.now()
    print('[+] ACTIVE & FACE & имя и описание норм', now.strftime("%Y-%m-%d %H:%M:%S") )
    print(s)
    print("--------------------------------------------------")
    log_description( s )
    good_woman_save()
    x, y = find_icon_on_screen("level2_yes.png")
    #x, y = find_icon_on_screen("level2_no.png") # for testing purposes
    pyautogui.click(x,y)
    good_profiles += 1


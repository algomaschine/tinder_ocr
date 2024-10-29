import pyperclip
from pynput import mouse

def on_click(x, y, button, pressed):
    if button == mouse.Button.middle and pressed:
        coords = f"{x}, {y}"
        pyperclip.copy(coords)
        print( coords)

with mouse.Listener(on_click=on_click) as listener:
    listener.join()

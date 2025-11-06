import warnings
warnings.filterwarnings("ignore")
import os
import tkinter as tk
from tkinter import filedialog, Text, END
from PIL import Image, ImageTk
from vehicle_detection import detect_vehicles, classify_vehicle
from attributes import classify_color, detect_license_plate, process_image

def run_gui():
    # opening file explorer to choose picture
    try:
        initial_dir=os.path.dirname(os.path.abspath(__file__))
    except Exception:
        initial_dir=os.getcwd()
    cars_dir=os.path.join(initial_dir,"cars")
    initial_dir=cars_dir if os.path.isdir(cars_dir) else initial_dir

    root=tk.Tk()
    root.withdraw()   # hide main window until a file is chosen
    root.update()     # ensure Tk is initialized before opening dialog

    path=filedialog.askopenfilename(
        parent=root,
        title="choose an image",
        initialdir=initial_dir,
        filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")]
    )

    # if cancelled — exit cleanly without opening a window
    if not path:
        root.destroy()
        return

    # show the results window only after a selection
    root.deiconify()
    root.title("Simple Vehicle Classifier- Results")
    root.geometry("500x430")

    # top bar: file path + choose another picture
    top_frame=tk.Frame(root)
    top_frame.pack(fill="x",padx=16,pady=8)

    path_var=tk.StringVar(value=path)
    lbl_path=tk.Label(
        top_frame,textvariable=path_var,
        font=("Arial",10),fg="#666",anchor="w",
        justify="left",wraplength=700
    )
    lbl_path.pack(side="left",fill="x",expand=True)

    btn_choose=tk.Button(top_frame,text="choose another picture",width=22)
    btn_choose.pack(side="right")

    # image display
    image_label=tk.Label(root)
    image_label.pack(pady=10)

    # output text box
    output_text=Text(root,font=("Arial",12),height=12)
    output_text.pack(padx=16,pady=10,fill="both",expand=True)

    def render(p):
        output_text.delete("1.0",END)
        try:
            process_image(p,output_text,image_label)
        except Exception as e:
            output_text.insert(END,f"Error: {e}\n")

    def choose_again():
        try:
            start_dir=os.path.dirname(path_var.get()) if path_var.get() else initial_dir
        except Exception:
            start_dir=initial_dir

        new_path=filedialog.askopenfilename(
            parent=root,
            title="choose an image",
            initialdir=start_dir,
            filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp")]
        )
        if not new_path:
            return
        path_var.set(new_path)
        render(new_path)

    btn_choose.config(command=choose_again)

    # initial run on the chosen file
    render(path)

    root.mainloop()

# optional no-op to keep compatibility if something still calls it
def browse_file(output_text=None,image_label=None):
    pass

if __name__=="__main__":
    run_gui()

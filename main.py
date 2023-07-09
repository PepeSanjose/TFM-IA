from tkinter import Tk, Label
import customtkinter
import winsound
from PIL import ImageTk, Image
import ProcesarVideo

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        #Configurar ventana ---------------------------------------------------------
        self.title("Sistema de detección de somnolencia")
        self.geometry("1000x500")
        self.grid_columnconfigure((0, 1), weight=1)

        #Imagen de fondo ------------------------------------------------------------
        self.imagen_fondo = Image.open("img/fondo.png")
        self.imagen_fondo = self.imagen_fondo.resize((self.winfo_screenwidth(), self.winfo_screenheight()), Image.ANTIALIAS)
        self.imagen_fondo_tk = ImageTk.PhotoImage(self.imagen_fondo)
        self.fondo = Label(self, image=self.imagen_fondo_tk)
        self.fondo.place(x=0, y=0, relwidth=1, relheight=1)

        #boton ---------------------------------------------------------------------
        self.button = customtkinter.CTkButton(self, text="Comenzar", command=self.button_callback)
        self.button.place(relx=0.5, rely=0.5, anchor=customtkinter.CENTER)

    def button_callback(self):
        self.button.place_forget()
        ProcesarVideo.mostrar_webcam()

app = App()
app.mainloop()


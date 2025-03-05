import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from utils.image_utils import predict_asl

class ASLDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sign Detection")
        self.root.geometry("400x400")

        # Add a button to upload an image
        self.upload_button = tk.Button(
            root, text="Upload Image", command=self.upload_image
        )
        self.upload_button.pack(pady=20)

        # Add a label to display the uploaded image
        self.image_panel = tk.Label(root)
        self.image_panel.pack()

        # Add a label to display the prediction result
        self.result_label = tk.Label(root, text="Predicted Sign: ", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def upload_image(self):
        """
        Handle image upload and display the prediction.
        """
        file_path = filedialog.askopenfilename()
        if file_path:
            # Display the uploaded image
            img = Image.open(file_path)
            img = img.resize((150, 150), Image.Resampling.LANCZOS)  # Updated line
            img = ImageTk.PhotoImage(img)
            self.image_panel.config(image=img)
            self.image_panel.image = img
    
            # Predict the ASL sign
            prediction = predict_asl(file_path)
            self.result_label.config(text=f"Predicted Sign: {prediction}")


if __name__ == "__main__":
    root = tk.Tk()
    app = ASLDetectionApp(root)
    root.mainloop()
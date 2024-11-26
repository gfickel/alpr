from pathlib import Path
import argparse

import numpy as np
import imgui
from PIL import Image
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import glfw
import cv2

from alpr import ALPR

class ALPRViewer:
    def __init__(self, model: ALPR):
        self.model = model
        self.current_image_path = None
        self.current_image = None
        self.current_results = None
        self.texture_id = None
        self.image_list = []
        self.current_image_idx = 0
        self.webcam_mode = False
        self.cap = None
        self.default_width = 1280
        self.default_height = 720

    
    def init_gui(self):
        if not glfw.init():
            return False
        
        # Create window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(1280, 720, "ALPR Viewer", None, None)
        if not self.window:
            glfw.terminate()
            return False

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Initialize ImGui
        imgui.create_context()
        self.impl = GlfwRenderer(self.window)
        
        # Set style
        style = imgui.get_style()
        style.window_rounding = 5.0
        style.frame_rounding = 3.0
        style.colors[imgui.COLOR_TEXT] = (0.90, 0.90, 0.90, 1.00)
        style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.06, 0.06, 0.06, 0.94)
        
        return True
    
    def update_texture(self):
        """Update OpenGL texture with current image and detections."""
        if self.current_image is None:
            return

        # Convert PIL image to numpy array
        img_array = np.array(self.current_image)
        
        # Draw detections
        if self.current_results is not None:
            bboxes, texts, kps_list = self.current_results
            for (l, t, r, b), text, kps in zip(bboxes, texts, kps_list):
                cv2.rectangle(img_array, (l, t), (r, b), (0, 255, 0), 2)
                # Draw keypoints
                for kp in kps:
                    x, y = int(kp[0]), int(kp[1])  # Convert to integers for cv2
                    cv2.circle(img_array, (x, y), 5, (0, 0, 255), -1)  # Red dots, radius 5, filled
                cv2.putText(img_array, text, (l, t-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Create OpenGL texture
        if self.texture_id is None:
            self.texture_id = gl.glGenTextures(1)
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, img_array.shape[1], img_array.shape[0],
                       0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, img_array)

    def init_webcam(self):
        """Initialize webcam capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return False
        self.webcam_mode = True
        self.load_current_image()
        return True

    def load_image_folder(self, folder_path: str):
        """Load all images from a folder."""
        if folder_path is None:
            return self.init_webcam()
            
        self.image_list = [
            str(p) for p in Path(folder_path).glob("*")
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
        ]
        if self.image_list:
            self.current_image_idx = 0
            self.load_current_image()
            return True
        return False

    def load_current_image(self):
        """Load and process current image."""
        if self.webcam_mode:
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(frame_rgb)
                self.current_results = self.model.forward(self.current_image)
                self.update_texture()
        elif 0 <= self.current_image_idx < len(self.image_list):
            self.current_image_path = self.image_list[self.current_image_idx]
            self.current_image = Image.open(self.current_image_path).convert('RGB')
            self.current_results = self.model.run_im_path(self.current_image_path)
            self.update_texture()

    def render_gui(self):
        """Render the GUI frame."""
        if self.current_image is None:
            return

        # Get current window size
        window_width, window_height = glfw.get_window_size(self.window)
        
        # Set ImGui window to match GLFW window size
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(window_width, window_height)
        
        # Window flags for fixed position but allowing resize through the OS window
        window_flags = (
            imgui.WINDOW_NO_MOVE | 
            imgui.WINDOW_NO_COLLAPSE | 
            imgui.WINDOW_NO_TITLE_BAR |
            imgui.WINDOW_NO_RESIZE
        )
        
        imgui.begin("ALPR Viewer", flags=window_flags)
        
        # Calculate layout
        panel_width = 200  # Width of the right panel
        padding = 10
        available_width = imgui.get_window_width() - panel_width - (3 * padding)
        
        # Begin main content area (left side)
        imgui.begin_child("left_panel", width=available_width, height=0, border=False)
        
        # Navigation controls (only show in image folder mode)
        if not self.webcam_mode:
            if imgui.button("Previous") and self.current_image_idx > 0:
                self.current_image_idx -= 1
                self.load_current_image()
            
            imgui.same_line()
            if imgui.button("Next") and self.current_image_idx < len(self.image_list) - 1:
                self.current_image_idx += 1
                self.load_current_image()

            imgui.text(f"Image: {self.current_image_path}")
        else:
            imgui.text("Webcam Mode")

        # Display image
        if self.current_image:
            image_width, image_height = self.current_image.size
            aspect_ratio = image_width / image_height
            display_width = min(available_width - padding, image_width)
            display_height = display_width / aspect_ratio

            imgui.image(self.texture_id, display_width, display_height)
        
        imgui.end_child()  # End left panel
        
        # Right panel for license plate information
        imgui.same_line(spacing=padding)
        imgui.begin_child("right_panel", width=panel_width, height=0, border=True)
        
        # Display results in right panel
        if self.current_results:
            bboxes, texts, kps = self.current_results
            imgui.text("Detected Licenses")
            imgui.separator()
            
            for i, text in enumerate(texts, 1):
                imgui.push_id(str(i))
                imgui.begin_child(f"license_{i}", height=60, border=True)
                imgui.text(f"Plate {i}")
                imgui.text_wrapped(text)
                imgui.end_child()
                imgui.spacing()
                imgui.pop_id()
        else:
            imgui.text_wrapped("No license plates detected")
            
        imgui.end_child()  # End right panel
        
        imgui.end()

        # In webcam mode, continuously update the image
        if self.webcam_mode:
            self.load_current_image()


    def run(self):
        """Main application loop."""
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()

            imgui.new_frame()
            self.render_gui()
            
            gl.glClearColor(0.1, 0.1, 0.1, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            imgui.render()
            self.impl.render(imgui.get_draw_data())
            
            glfw.swap_buffers(self.window)

        # Cleanup
        if self.webcam_mode and self.cap is not None:
            self.cap.release()
        self.impl.shutdown()
        glfw.terminate()


def get_args():
    parser = argparse.ArgumentParser(description='ALPR with GUI viewer')
    parser.add_argument('--detector_model', required=True, help='Detector binary path')
    parser.add_argument('--ocr_model', required=True, help='OCR binary path')
    parser.add_argument('--ocr_config', required=True, help='OCR Model config path')
    parser.add_argument('--image_folder', help='Folder containing images to process. If not provided, uses webcam.')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Initialize model
    model = ALPR(args.detector_model, args.ocr_model, args.ocr_config)
    
    # Initialize and run viewer
    viewer = ALPRViewer(model)
    if viewer.init_gui():
        viewer.load_image_folder(args.image_folder)
        viewer.run()

class Config:
    def __init__(self):
        # --- General Settings ---
        self.enableaim = True
        self.enabletb = False
        self.offsetX = 0
        self.offsetY = 5

        self.color = "purple"
        self.language = "english"
        
        # --- Mouse / MAKCU ---
        self.selected_mouse_button = 1
        self.selected_sec_mouse_button = 2
        self.selected_tb_btn = 1
        self.in_game_sens = 0.235
        self.mouse_dpi = 800
        
        # --- Aimbot Mode ---
        self.mode = "Normal"    

        # --- FOV Settings ---
        self.fovsize = 100
        self.fov_color = "white"
        self.fov_smooth_color = "cyan"
        
        # --- Triggerbot Settings ---
        self.tbfovsize = 5 
        self.tbdelay = 0.5
        self.tbcooldown = 0.5
        self.tb_fov_color = "white"
        
        # --- Main Aimbot Button ---
        self.main_x_speed = 3
        self.main_y_speed = 3
        
        # --- Secondary Aimbot Button ---
        self.sec_x_speed = 1.5
        self.sec_y_speed = 1.5
        
        # --- For backward compatibility ---
        self.normal_x_speed = self.main_x_speed
        self.normal_y_speed = self.main_y_speed

        # --- Common Aim Settings ---
        self.normalsmooth = 30
        self.normalsmoothfov = 30
    

config = Config()
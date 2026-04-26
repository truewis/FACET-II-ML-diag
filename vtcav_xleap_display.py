import numpy as np
import time
import epics
from scipy.fft import fft2, ifft2, ifftshift
from scipy.ndimage import label, center_of_mass, mean as nd_mean
from qtpy.QtCore import QThread, Signal, Slot
from vtcav_display import VTCAVDisplay
LCLS_XTCAV_PROF_MON_PV = 'OTRS:DMPH:695:Image'
# ==========================================
# 1. Background watcherWorker Class
# ==========================================
class ProfMonWatcherWorker(QThread):
    # Define signals to send data back to the main UI thread
    xleap_score_computed = Signal(float)
    image_processed = Signal(np.ndarray)
    new_log_signal = Signal(str)
    display_images_rt = True  # Control flag for real-time image display
    def __init__(self, pv_name, width, parent=None):
        super().__init__(parent)
        self.pv_name = pv_name
        self.syag_width = width
        self.running = False
        # Algorithm parameters
        self.sigma_x = 3.2
        self.sigma_y = 0.1
        self.balance = 0.05
        self.window_size = 5
    def _log(self, message):
        #print(f"[Watcher] {message}")
        self.new_log_signal.emit(f"[Watcher] {message}")
    def run(self):
        """
        This method executes in the background when self.start() is called.
        """
        self.running = True
        while self.running:
            start_time = time.time()
            # 1. Fetch image
            syag_data = epics.PV(self.pv_name + ":ArrayData").get()
            if syag_data is None or self.syag_width is None:
                return

            # 2. Reshape and crop
            syag_image = np.array(syag_data).reshape(-1, self.syag_width).T
            # 1. Background thresholding on a copy to preserve original image data for the crop
            temp_image = syag_image.copy()
            median_val = np.median(temp_image)
            temp_image[temp_image < median_val * 3] = 0

            # 2. Find clusters of adjacent non-zero pixels
            mask = temp_image > 0
            labeled_array, num_features = label(mask)

            if num_features > 0:
                # 3. Find the cluster with the highest mean intensity
                # nd_mean efficiently calculates the mean of temp_image for each labeled cluster
                indices = np.arange(1, num_features + 1)
                cluster_means = nd_mean(temp_image, labels=labeled_array, index=indices)
                best_cluster_idx = indices[np.argmax(cluster_means)]

                # 4. Compute Center of Mass for the best cluster
                com_y, com_x = center_of_mass(temp_image, labels=labeled_array, index=best_cluster_idx)
                com_y, com_x = int(np.round(com_y)), int(np.round(com_x))
            else:
                # Fallback if the image is completely blank/dark
                com_y, com_x = syag_image.shape[0] // 2, syag_image.shape[1] // 2

            # 5. Define fixed rectangle target dimensions
            crop_h, crop_w = 400, 150
            half_h, half_w = crop_h // 2, crop_w // 2

            # Calculate raw crop boundaries
            y_start, y_end = com_y - half_h, com_y + half_h
            x_start, x_end = com_x - half_w, com_x + half_w

            # 6. Safe Cropping: Pad the original image if the crop window goes out of bounds
            pad_top = max(0, -y_start)
            pad_bottom = max(0, y_end - syag_image.shape[0])
            pad_left = max(0, -x_start)
            pad_right = max(0, x_end - syag_image.shape[1])

            # Pad with zeros (background) to satisfy the 400x150 requirement
            padded_syag = np.pad(syag_image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

            # Adjust slice coordinates based on the padding we just added
            y_start_p, y_end_p = y_start + pad_top, y_end + pad_top
            x_start_p, x_end_p = x_start + pad_left, x_end + pad_left

            cropped_syag = padded_syag[y_start_p:y_end_p, x_start_p:x_end_p]

            # 3. Generate PSF & Deconvolve
            psf = self.generate_anisotropic_psf(cropped_syag.shape, self.sigma_x, self.sigma_y)
            recovered_img = self.wiener_deconvolution(cropped_syag, psf, self.balance)

            # 4. Thresholding based on median background
            median_bg = np.median(recovered_img)
            recovered_img[recovered_img < median_bg * 3] = 0

            # 5. Find peak position (assuming column projection)
            recovered_cp = np.sum(recovered_img, axis=0) 
            peak_pos = np.argmax(recovered_cp)

            # 6. Compute Spread
            result = self.compute_weighted_y_variance_window(recovered_img, self.window_size)

            try:
                energy_spread = result['window_scores'][peak_pos]
            except IndexError:
                energy_spread = 0.0

            # 7. Emit results to GUI
            self.xleap_score_computed.emit(energy_spread)
            if self.display_images_rt:
               self.image_processed.emit(recovered_img)
               
            # Maintain requested update rate (10Hz)
            elapsed = time.time() - start_time
            if elapsed > 0.1:
                self._log(f"Warning: Prediction Thread Cannot Catch up at 10 Hz. Time took for prediction: {elapsed}.")
            time.sleep(max(0.01, 0.1 - elapsed))

    # --- Processing Methods ---
    def generate_anisotropic_psf(self, shape, sigma_x, sigma_y):
        rows, cols = shape
        x = np.arange(-cols // 2, cols // 2) + (cols % 2) / 2.0
        y = np.arange(-rows // 2, rows // 2) + (rows % 2) / 2.0
        X, Y = np.meshgrid(x, y)
        psf = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
        psf /= np.sum(psf)
        return psf

    def wiener_deconvolution(self, blurred_image, psf, balance):
        psf_shifted = ifftshift(psf)
        IMG_FFT = fft2(blurred_image)
        PSF_FFT = fft2(psf_shifted)
        PSF_FFT_conj = np.conj(PSF_FFT)
        psf_power = np.abs(PSF_FFT)**2
        wiener_filter = PSF_FFT_conj / (psf_power + balance)
        recovered_fft = IMG_FFT * wiener_filter
        return np.real(ifft2(recovered_fft))

    def compute_weighted_y_variance_window(self, image, window_size):
        rows, cols = image.shape
        y_coords = np.arange(rows).reshape(-1, 1)
        
        S0 = np.sum(image, axis=0) 
        S1 = np.sum(y_coords * image, axis=0)
        S2 = np.sum((y_coords**2) * image, axis=0)
        
        sliding_window = np.ones(window_size)
        M0 = np.convolve(S0, sliding_window, mode='valid')
        M1 = np.convolve(S1, sliding_window, mode='valid')
        M2 = np.convolve(S2, sliding_window, mode='valid')
        
        M0_safe = M0 + 1e-10
        expected_y = M1 / M0_safe
        expected_y2 = M2 / M0_safe
        weighted_variance = expected_y2 - (expected_y ** 2)
        
        max_index = np.argmax(weighted_variance)
        return {
            'max_index': max_index,
            'max_variation': weighted_variance[max_index],
            'window_scores': weighted_variance
        }
    
    def stop(self):
        self.running = False
        self.wait()

# ==========================================
# 2. Main PyDM Display Class
# ==========================================
class VTCAVDisplay_XLEAP(VTCAVDisplay):
    def __init__(self, parent=None, args=None):
        self.display_mapping = {
                'RT':   ('imageContainer_1', 'currentProfile_1', 'energyProfile_1'),
                'DAQ':  ('imageContainer_2', 'currentProfile_2', 'energyProfile_2'),
                'Prep': ('imageContainer_3', 'currentProfile_3', 'energyProfile_3'),
                'cmp_truth': ('imageContainer_4', 'currentProfile_4', 'energyProfile_4'),
                'cmp_pred': ('imageContainer_5', 'currentProfile_5', 'energyProfile_5'),
                'xleap': ('imageContainer_7', 'currentProfile_7', 'energyProfile_7'),
            }
        super().__init__(parent=parent, args=args)
        
        # Initialize the watcherWorker
        # Replace 'YOUR_SYAG_PV' and width with your actual parameters
        self.watcherWorker = ProfMonWatcherWorker(pv_name=LCLS_XTCAV_PROF_MON_PV, width=epics.PV(LCLS_XTCAV_PROF_MON_PV+":ArraySize0_RBV").get(), parent=self)
        self.watcherWorker.xleap_score_computed.connect(self.update_xleap_score)
        self.watcherWorker.new_log_signal.connect(self.handle_log)
        self.watcherWorker.image_processed.connect(lambda data: self.update_image_display(data, display_name='xleap'))
        # Connect watcherWorker signals to UI update functions
        self.setup_connections_xleap()
        
        # Example: Trigger watcherWorker on a button click
        # self.ui.calc_spread_btn.clicked.connect(self.start_watcherWorker)

    def ui_filename(self):
        return 'vtcav_xleap_display.ui'
    
    def setup_connections_xleap(self):
        # Control
        self.ui.startPauseButton_xleap.clicked.connect(self.toggle_acquisition_xleap)
        self.ui.doImage_xleap.stateChanged.connect(self.toggle_doImage_xleap)
    def start_watcherWorker(self):
        """Starts the background thread if it isn't already running."""
        if not self.watcherWorker.isRunning():
            self.watcherWorker.start()
    @Slot()
    def toggle_acquisition_xleap(self):
        """Toggles between Start and Pause states using the pre-loaded worker."""
        btn = self.ui.startPauseButton_xleap
        
        if self.watcherWorker is None:
            self.handle_log("Error: watcher worker failed to initialize. Try restarting the application.")
            return

        # Check if worker is running
        if self.watcherWorker.isRunning():
            # PAUSE
            self.watcherWorker.stop()
            btn.setText("Start")
            self.handle_log("XLEAP Acquisition Paused")
        else:
            # START
            btn.setText("Pause")
            self.handle_log("XLEAP Acquisition Started")
            self.watcherWorker.start()

    def update_xleap_score(self, data):
        color_style = "color:#ff0000;" if data < 50 else ""
        html = f'<html><head/><body><p><span style="font-size:48pt; font-weight:600; {color_style}">{int(data)} %</span></p></body></html>'
        self.ui.scoreLabel_xleap.setText(html)

    @Slot(int)
    def toggle_doImage_xleap(self, state):
        if self.watcherWorker is None:
            return
        if state == 0:
            self.watcherWorker.display_images_rt = False
        else:
            self.watcherWorker.display_images_rt = True

# Entry point for PyDM
intelclass = VTCAVDisplay_XLEAP

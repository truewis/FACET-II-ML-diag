import os
import time
import pickle
import numpy as np
import torch
import epics
import glob
import sys
import traceback
import re
import matplotlib
import pathlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from Python_Functions.cvae import CVAE, vae_loss, smooth_cvae_output
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from scipy.io import loadmat
from scipy.ndimage import center_of_mass, shift
from Python_Functions.functions import cropProfmonImg, matstruct_to_dict, extractDAQBSAScalars, extractDAQNonBSAScalars, segment_centroids_and_com, apply_tcav_zeroing_filter, apply_centroid_correction, extract_processed_images
from qtpy.QtGui import QColor
from qtpy.QtCore import QThread, Signal, Slot, Qt
from qtpy.QtWidgets import QMessageBox, QFileDialog, QTableWidgetItem
from pydm import Display
import pyqtgraph as pg

# Constants for FACET-II/BSA Timing
PID_MASK = 0x1FFFF  
PID_MODULUS = 0x20000 
BSA_BUFFER_LENGTH = 2800 
EMPTY_BUFFER = np.zeros(BSA_BUFFER_LENGTH)
CHARGE_PV_C = 'TORO:LI20:2452:TMIT'
CHARGE_PV_U = 'TORO_LI20_2452_TMIT'

# DAQ Constants
DAQPATH = "/nas/nas-li20-pm00/"
sys.path.append("/usr/local/facet/tools/python")
from F2_pytools.f2bsaBuffer import f2BeamSynchronousBuffer
from F2_pytools.controls_jurisdiction import is_SLC

class InferenceWorker(QThread):
    new_prediction_signal = Signal(np.ndarray)
    new_pv_values = Signal(np.ndarray, np.ndarray)
    new_log_signal = Signal(str)
    
    def __init__(self, model_path):
        print("Initializing Inference Worker at "+model_path)
        super().__init__()
        self.model_path = model_path
        self.running = False
        self.model_data = None
        
        # Load resources and initialize the Buffer class
        self.load_model_resources()

    def load_model_resources(self):
        if not os.path.exists(self.model_path):
            self._log(f"Error: Model file not found at {self.model_path}")
            return

        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)

            self.model = data['model']
            self.x_scaler = data['x_scaler']
            self.iz_scaler = data['iz_scaler']
            self.image_model = data['image_model']
            try:
                self.xtcalibrationfactor = data['xtcalibrationfactor']
            except:
                self.xtcalibrationfactor = 3.2 # known rough value for FACET-II
                self._log("Warning: No xtcalibrationfactor found in the file.")
            #The following solution replaces every _ with :, unless it is immediately preceded by the exact token "FAST", i.e. FAST_PACT
            #This deals with the absolutely crazy pv name formatting of F2_DAQ.
            self.var_names = [re.sub(r'(?<!^FAST)(?<!_FAST)_', ':', name) for name in data['varNames']]
            self.bsa_names = [re.sub(r'(?<!^FAST)(?<!_FAST)_', ':', name) for name in data['bsaVarNames']]
            self.nonbsa_names = [re.sub(r'(?<!^FAST)(?<!_FAST)_', ':', name) for name in data['nonBsaVarNames']]
            self.is_pv_bypassed = [False]*len(self.var_names)
            arch = data.get('architecture', {})
            self.ncomp = arch.get('ncomp', 1)

            # Create Sync Buffer for the first time.
            self.create_sync_buffer()
            
            self._log(f"Model loaded. EPICS: {len(self.epics_list)}, SCP: {len(self.scp_list)}")
        except Exception as e:
            traceback.print_exc()
            self._log(f"Failed to load model resources: {e}")
    def create_sync_buffer(self):
        self._log("Creating Sync Buffer. This might take a while......")
        # --- Distinguish between SCP and EPICS ---
        self.epics_list = []
        self.scp_list = []
        bsa_intersection = [n for n, bypassed in zip(self.var_names, self.is_pv_bypassed) if n in self.bsa_names and not bypassed]
        for name in bsa_intersection:
            # Remove TMIT/X/Y suffixes for jurisdiction check if necessary, 
            # but is_SLC usually handles the base device name.
            try:
                if is_SLC(name):
                    # scp list at $TOOLS/python/F2_pytools/controls_jurisdictions.csv has not been updated for a while.
                    # hence scp acquire does not work. Try to read out everything in epics.
                    # self.scp_list.append(name)
                    self.epics_list.append(name)
                else:
                    self.epics_list.append(name)
            except:
                self.epics_list.append(name)
        self.sync_buffer = f2BeamSynchronousBuffer(
                EPICS_address_list=self.epics_list,
                SCP_device_list=self.scp_list,
                SCP_Npts=100, # Minimal buffer for SCP to stay fast
                verbose=True,
                nowait=True #Don't wait for SCP, as it doesn't work right now anyways.
            )

    def _log(self, message):
        print(f"[Worker] {message}")
        #self.new_log_signal.emit(f"[Worker] {message}")
        
    def run(self):
        self.running = True
        self._log("Inference Worker Started (Hybrid Mode)")

        # Initialize non-BSA PV objects once
        non_bsa_pvs = {}
        for name in self.var_names:
            if name not in self.bsa_names:
                non_bsa_pvs[name] = epics.PV(name)

        # Track previous valid BSA list to detect changes
        previous_valid_bsa_names = []

        while self.running:
            start_time = time.time()

            try:
                # --- Phase 1: Read Synchronized Data (BSA) ---
                # We only try to read from the buffer if we have valid names in it
                latest_sync_shot = None
                valid_bsa_names = [name for i, name in enumerate(self.var_names) 
                                   if name in self.bsa_names and not self.is_pv_bypassed[i]]
                
                # Dynamic Buffer Re-creation: If the list of valid BSA PVs changed, rebuild buffer
                #Unbypassed n/a pv variables causes timeout, which we don't want because it slows down the sync buffer.
                if valid_bsa_names != previous_valid_bsa_names:
                    self._log(f"Rebuilding BSA Buffer. Active PVs: {len(valid_bsa_names)}")
                    self.create_sync_buffer()
                    previous_valid_bsa_names = valid_bsa_names

                # Get data if we have valid PVs to read
                if valid_bsa_names:
                    sync_namelist, sync_data = self.sync_buffer.get_data()
                    
                    if sync_data is not None and sync_data.shape[1] > 0:
                        # Take the last column (latest shot)
                        latest_sync_shot = sync_data[:, -1]
                    else:
                        self._log("Warning: Sync buffer returned no data.")

                # --- Phase 2: Build Input Vector & Auto-Bypass ---
                ordered_input = []
                
                for i, name in enumerate(self.var_names):
                    val = None
                    
                    # Skip readout if already bypassed                
                    if self.is_pv_bypassed[i]:
                        # Already bypassed, use placeholder
                        val = 0.0 
                    else:
                        # Attempt Readout
                        if name in self.bsa_names:
                            # It should be in our sync_namelist if it wasn't bypassed before
                            if latest_sync_shot is not None and name in sync_namelist:
                                idx = sync_namelist.index(name)
                                val = latest_sync_shot[idx]
                            else:
                                val = None # buffer read failed or empty
                        else:
                            # Non-BSA Readout
                            val = non_bsa_pvs[name].get()

                    # Check for Validity
                    if val is None or np.isnan(val):
                        if not self.is_pv_bypassed[i]:
                            self._log(f"Recommend Bypassing PV: {name} (Value: {val})")
                            self.is_pv_bypassed[i] = 1 # Set flag
                        ordered_input.append(0.0) # Placeholder for raw data
                    else:
                        ordered_input.append(val)

                # Convert to numpy array
                input_array = np.array(ordered_input).reshape(1, -1)
                self.new_pv_values.emit(input_array[0], np.array(self.is_pv_bypassed))
                # --- Phase 3: Scale & Force Zeros ---
                # 1. Transform raw inputs
                input_scaled = self.x_scaler.transform(input_array)
                
                # 2. Force bypassed indices to exactly 0.0 in the SCALED vector.
                for i, is_bypassed in enumerate(self.is_pv_bypassed):
                    if is_bypassed:
                        input_scaled[0, i] = 0.0

                # --- Phase 4: Handle Charge (TORO) ---
                # We need the raw charge for display/logging
                toro_pv = CHARGE_PV_C
                total_charge = 0.0
                if toro_pv in self.var_names:
                    idx = self.var_names.index(toro_pv)
                    # Use the raw value we collected, or 0 if it was bypassed
                    total_charge = ordered_input[idx]
                
                # --- Phase 5: Inference ---
                self.emit_prediction(input_scaled, total_charge)

            except Exception as e:
                self._log(f"Loop Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)

            # Maintain requested update rate (~2Hz)
            elapsed = time.time() - start_time
            time.sleep(max(0.01, 0.5 - elapsed))

    def emit_prediction(self, input_params, total_charge):
        try:
            X_test = torch.tensor(input_params, dtype=torch.float32)
            
            if hasattr(self.model, 'eval'): self.model.eval()
            with torch.no_grad():
                z_pred_full = self.model.predict(X_test)

            pred_full = self.iz_scaler.inverse_transform(z_pred_full)
            pred_params = pred_full.flatten()
            print(pred_params)
            
            image_data = self.image_model.params_to_image(pred_params, total_charge)
            # Figuring out orientation sucks. Seems to work without a T, but sometimes with a T.
            self.new_prediction_signal.emit(image_data)
            self._log(f"New Prediction Emitted. Charge: {total_charge*1.6e-7} pC")
        except Exception as e:
            raise e
            self._log(f"Prediction Error: {e}")

    def stop(self):
        self.running = False
        self.wait()

        
class VTCAVDisplay(Display):
    def __init__(self, parent=None, args=None):
        super().__init__(parent=parent, args=args)
        
        self.model_folder = "./models/"
        self.current_model_path = ""
        self.worker = None
        self.predictors = None
        self.predictor_vars = None

        # 1. Setup UI Elements
        self.setup_model_ui()
        self.setup_daq_ui()
        self.setup_connections()
        
        # Initialize plots
        self.init_plots()
        self.setup_image_plot()

    def ui_filename(self):
        return "vtcav_display.ui"

    def setup_model_ui(self):
        """Populate modelName combobox with files from ./models/"""
        if os.path.exists(self.model_folder):
            files = glob.glob(os.path.join(self.model_folder, "*.pkl"))
            filenames = [os.path.basename(f) for f in files]
            self.ui.modelName.addItems(filenames)
        else:
            self.handle_log(f"Warning: Model directory {self.model_folder} does not exist.")

    def setup_daq_ui(self):
        """This function is useless for now."""
        if os.path.exists(DAQPATH):
            try:
                pass
            except Exception as e:
                self.handle_log(f"Error reading DAQ path: {e}")
        else:
            self.handle_log(f"Warning: DAQ Path {DAQPATH} does not exist.")

    def setup_connections(self):
        # Model Loading
        self.ui.modelLoadButton.clicked.connect(self.load_selected_model)
        
        # DAQ Interaction
        self.ui.daqLoadButton.clicked.connect(self.handle_daq_load)

        # DAQ Shot Navigation: nextShotButton, prevShotButton, shotNumberSlider
        self.ui.nextShotButton.clicked.connect(self.daqNextShot)
        self.ui.prevShotButton.clicked.connect(self.daqPrevShot)
        self.ui.shotNumberSlider.valueChanged.connect(self.emit_daq_image) # Reload data when shot number changes
        # Control
        self.ui.startPauseButton.clicked.connect(self.toggle_acquisition)

        # Table intereaction
        self.ui.pvTable.cellClicked.connect(self.handle_table_click)

        # Tab 6: Preprocess XTCAV Image - Load Raw DAQ
        self.ui.tcavDaqLoadButton.clicked.connect(self.handle_tcav_daq_load)

        # Tab 6: Preprocess XTCAV Image - Write Processed File
        # This handler would likely read the ROI/Range inputs (xroiMin, xroiMax, etc.)
        self.ui.preprocessWriteButton.clicked.connect(self.handle_preprocess_write)

        # Tab 5: Train New Model - Load Preprocessed Data
        self.ui.preprocessLoadButton.clicked.connect(self.handle_preprocess_load)

        # Tab 5: Train New Model - Train & Write Model File
        self.ui.modelWriteButton.clicked.connect(self.handle_model_train_write)
    @Slot()
    def daqNextShot(self):
        # increase the shot slider by 1, ensuring it doesn't exceed max
        current_value = self.ui.shotNumberSlider.value()
        max_value = self.ui.shotNumberSlider.maximum()
        if current_value < max_value:
            self.ui.shotNumberSlider.setValue(current_value + 1)
    @Slot()
    def daqPrevShot(self):
        # decrease the shot slider by 1, ensuring it doesn't go below min
        current_value = self.ui.shotNumberSlider.value()
        min_value = self.ui.shotNumberSlider.minimum()
        if current_value > min_value:
            self.ui.shotNumberSlider.setValue(current_value - 1)
    
    @Slot()
    def emit_daq_image(self):
        # This function will be called when the shot number slider changes.
        # It should load the corresponding shot data and emit it to the image display.
        if self.predictors is None or self.predictor_vars is None:
            # No data loaded yet
            return
        index = self.ui.shotNumberSlider.value()
        self.ui.shotNumber.setText(str(index)+" / "+str(self.ui.shotNumberSlider.maximum()))
        # Find total charge for this shot by looking up for TORO:LI20:2452:TMIT in predictor_vars
        total_charge = self.predictors[index, self.predictor_vars.index(CHARGE_PV_C)] if CHARGE_PV_C in self.predictor_vars else None
        # Filter predictors by variable names, given in self.worker.var_names, to ensure correct ordering and selection
        # Find indices of predictor_vars in worker.var_names
        if self.worker is None or not hasattr(self.worker, 'var_names'):
            return
        var_indices = []
        for var in self.predictor_vars:
            try:
                idx = self.worker.var_names.index(var)
                var_indices.append(idx)
            except ValueError:
                #There could be more pvs than the model requires. Just ignore them.
                #self.handle_log(f"Warning: Variable {var} not found in model var_names.")
                continue
        # Extract the corresponding predictor data for the selected shot
        # Array contains a single sample, hence reshape.
        filtered_predictor = self.predictors[index, var_indices].reshape(1, -1)
        scaled_predictor = self.worker.x_scaler.transform(filtered_predictor)
        self.worker.emit_prediction(scaled_predictor, total_charge)
        self.update_pv_values(filtered_predictor[0], np.array(self.worker.is_pv_bypassed))

    def init_plots(self):
        # Configure initial plot settings if necessary
        # PyDMEventPlot uses pyqtgraph's PlotItem internally
        pass

    @Slot()
    def handle_daq_load(self):
        """
        Opens a file dialog to select a DAQ file. 
        If selected, updates the text field and triggers loading.
        """
        # Open File Dialog starting at DAQPATH
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select DAQ Data File", 
            DAQPATH, 
            "All Files (*)"
        )

        if file_path:
            # Update the UI LineEdit to show the path
            if hasattr(self.ui, 'daqFilePath'):
                self.ui.daqFilePath.setText(file_path)
            
            # Call the loading function
            self.load_daq_data(file_path)
    def load_daq_data(self, full_path):
        # First, pause the real-time acquisition if it's running to avoid conflicts
        if self.worker and self.worker.isRunning():
            self.handle_log("Pausing acquisition to load DAQ data...")
            self.toggle_acquisition() # This will pause the worker
        self.handle_log(f"DAQ Load requested for: {full_path}")
        dataloc = full_path
        try:
            mat = loadmat(dataloc,struct_as_record=False, squeeze_me=True)
            data_struct = mat['data_struct']
        except FileNotFoundError:
            print(f"Skipping {experiment}_{runname}: .mat file not found at {dataloc}")
        # 2. Extract full BSA scalars (filtered by step_list if needed)
        # Don't filter by common index here, we'll do it with the goodShots scalar common index loaded from the file
        bsaScalarData, bsaVars = extractDAQBSAScalars(data_struct, filter_index=False)
        bsaScalarData = apply_tcav_zeroing_filter(bsaScalarData, bsaVars)
        # 3. Extract non BSA scalars the same way
        nonBsaScalarData, nonBsaVars = extractDAQNonBSAScalars(data_struct, filter_index=False, debug=False, s20=True)
        nonBsaScalarData = apply_tcav_zeroing_filter(nonBsaScalarData, nonBsaVars)

        # 4. Combine BSA and non-BSA scalar data
        bsaScalarData = np.vstack((bsaScalarData, nonBsaScalarData))
        bsaVars = bsaVars + nonBsaVars
        # 5. Filter BSA data using the final index
        # goodShots_scal_common_index is 1 based indexing from MATLAB, convert to 0 based
        bsaScalarData_filtered = bsaScalarData
        
        # 6. Construct the predictor array
        predictor_current = np.vstack(bsaScalarData_filtered).T
        all_predictors = []
        # C. Append to master lists (Some nonsensical code, but we want to maintain the structure in the notebook for now)
        all_predictors.append(predictor_current)
        predictor_tmp = np.concatenate(all_predictors, axis=0)
        self.predictors = predictor_tmp
        self.predictor_vars = [re.sub(r'(?<!^FAST)(?<!_FAST)_', ':', name) for name in bsaVars]
        self.handle_log(f"Loaded DAQ data with shape {predictor_current.shape}")
        self.ui.shotNumberSlider.setMaximum(predictor_current.shape[0])
        self.ui.shotNumberSlider.setMinimum(0)

    def setup_pv_table(self):
        """
        Populates the PV Table with static model information:
        - PV Name
        - Model Min (0 scaled to original)
        - Model Max (1 scaled to original)
        """
        # Ensure worker and model data exist
        if not self.worker or not hasattr(self.worker, 'x_scaler'):
            return

        # Get variable names and scaler
        var_names = self.worker.var_names
        scaler = self.worker.x_scaler
        
        # Calculate Min/Max by inverse transforming 0 and 1
        # We create dummy vectors of 0s and 1s to feed the scaler
        n_features = len(var_names)
        print(f"Number of Features:{n_features}")
        # Create input arrays for 0 (Min) and 1 (Max)
        # Note: We reshape to (1, -1) because scaler expects 2D array
        zeros = np.zeros((1, n_features))
        ones = np.ones((1, n_features))
        
        # Inverse transform to get physical values
        # This assumes MinMaxScaler. If StandardScaler, 0/1 are Z-scores.
        real_min = scaler.inverse_transform(zeros)[0]
        real_max = scaler.inverse_transform(ones)[0]

        # Configure Table
        self.ui.pvTable.setRowCount(n_features)
        self.ui.pvTable.setColumnCount(5) # Name, Min, Max, Value, Bypassed
        self.ui.pvTable.setHorizontalHeaderLabels([
            "PV Name", "Model Min", "Model Max", "Current Value", "Bypassed"
        ])

        # Populate Static Data
        for row, name in enumerate(var_names):
            # 1. PV Name
            self.ui.pvTable.setItem(row, 0, QTableWidgetItem(name))

            if name == CHARGE_PV_C:
                item_charge = self.ui.pvTable.item(row, 0)
                item_charge.setText(f"Charge({name})")
                item_charge.setBackground(QColor("blue"))
                item_charge.setForeground(QColor("white"))
        
            # 2. Model Min (Format to 4 decimals)
            min_item = QTableWidgetItem(f"{real_min[row]:.4e}")
            self.ui.pvTable.setItem(row, 1, min_item)
            
            # 3. Model Max
            max_item = QTableWidgetItem(f"{real_max[row]:.4e}")
            self.ui.pvTable.setItem(row, 2, max_item)
            
            # Initialize other columns to empty or default
            self.ui.pvTable.setItem(row, 3, QTableWidgetItem("-"))
            self.ui.pvTable.setItem(row, 4, QTableWidgetItem("No"))

        # Optional: Resize columns to content
        self.ui.pvTable.resizeColumnsToContents()


    def handle_tcav_daq_load(self):
        """
        Opens a file dialog to select the XTCAV DAQ file.
        If selected, updates the text field and triggers loading logic.
        """
        # Open File Dialog starting at DAQPATH (assuming DAQPATH is defined globally or in self)
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select XTCAV DAQ File", 
            DAQPATH, 
            "All Files (*)"
        )

        if file_path:
            # Update the UI LineEdit to show the path
            if hasattr(self.ui, 'tcavDaqFilePath'):
                self.ui.tcavDaqFilePath.setText(file_path)
            
            # Call the loading function to populate ROIs
            self.load_tcav_daq_data(file_path)

    def load_tcav_daq_data(self, file_path):
        """
        Loads the .mat file metadata to populate default ROI and Range text boxes.
        """
        try:
            # 1. Load just the header/struct to guess dimensions or set defaults
            # (In a real app, you might load the first image to find the beam center here)
            # For now, we populate with the defaults observed in your notebook 
            # or reasonable starting values.
            
            # Defaults observed in 2D_LPS_preprocess.ipynb:
            default_x_roi = (400, 700)
            default_y_roi = (400, 600)
            
                       
            # "xrangeTimesTwo and yrangeTimesTwo must be 200 by default"
            self.ui.xrangeTimesTwo.setPlainText("200")
            self.ui.yrangeTimesTwo.setPlainText("200")
            
            print(f"Loaded XTCAV DAQ: {file_path}. ROIs populated.")
            mat = loadmat(file_path,struct_as_record=False, squeeze_me=True)
            data_struct = mat['data_struct']
            hotPixThreshold = 1e3
            sigma = 1
            threshold = 5
            file_path_obj = pathlib.Path(file_path)
            xtcavImages_centroid_uncorrected, xtcavImages_raw, horz_proj, LPSImage = extract_processed_images(data_struct, '', 100, 100, hotPixThreshold, sigma, threshold, [1], None, None, do_load_raw=True, directory_path=str(file_path_obj.parent))# do_load_raw = False by default.
            # Populate UI elements
            dims = xtcavImages_raw.shape
            self.ui.xroiMin.setPlainText(str(0))
            self.ui.xroiMax.setPlainText(str(dims[1]))
            
            self.ui.yroiMin.setPlainText(str(0))
            self.ui.yroiMax.setPlainText(str(dims[0]))
            self.update_image_display(np.average(xtcavImages_raw, axis = 2))
            
        except Exception as e:
            print(f"Error initializing data from {file_path}: {e}")

    def handle_preprocess_write(self):
        """
        Reads ROI/Range parameters from UI, asks for save location, 
        and calls the processing function.
        """
        # 1. Get Input File Path
        thedaq_file_path = self.ui.tcavDaqFilePath.text()
        if not thedaq_file_path:
            print("Error: No DAQ file selected.")
            return

        # 2. Read ROI and Range values from TextBoxes
        try:
            x_min = int(self.ui.xroiMin.toPlainText())
            x_max = int(self.ui.xroiMax.toPlainText())
            y_min = int(self.ui.yroiMin.toPlainText())
            y_max = int(self.ui.yroiMax.toPlainText())
            
            # Note: The UI says "TimesTwo" (200), but the function usually takes 
            # the half-range (100) as 'xrange'. 
            x_range_val = int(self.ui.xrangeTimesTwo.toPlainText()) // 2
            y_range_val = int(self.ui.yrangeTimesTwo.toPlainText()) // 2
            
        except ValueError:
            print("Error: Please ensure all ROI and Range fields contain valid integers.")
            return

        # 3. Open Save Dialog
        filename = pathlib.Path(thedaq_file_path).name.replace(".mat", "_preprocessed.pkl") # Suggest a filename
        theoutput_file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Preprocessed Data",
            filename,
            "Pickle Files (*.pkl)"
        )

        if theoutput_file_path:
            self.ui.preprocessWritePath.setText(theoutput_file_path)
            
            print("Starting preprocessing...")
            
            # 4. Call the processing function defined earlier
            # Function signature: (daq_file_path, x_roi, y_roi, xrange, yrange, output_file_path)
            self.preprocess_and_save_lps(
                daq_file_path=thedaq_file_path,
                x_roi=(x_min, x_max),
                y_roi=(y_min, y_max),
                xrange=x_range_val,
                yrange=y_range_val,
                output_file_path=theoutput_file_path
            )
            print("Preprocessing Complete.")
                
    def handle_preprocess_load(self):
        """
        Opens a file dialog to select a preprocessed pickle file.
        Updates the UI to show the selected file path.
        """
        # Open File Dialog starting at a default processed data path if available
        # We'll use DAQPATH as a base or current directory
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Preprocessed Data File", 
            ".", 
            "Pickle Files (*.pkl)"
        )

        if file_path:
            # Update the UI LineEdit
            if hasattr(self.ui, 'preprocessFilePath'):
                self.ui.preprocessFilePath.setText(file_path)
            
            print(f"Selected preprocessed file: {file_path}")

    def handle_model_train_write(self):
        """
        Triggers the model training/aggregation process.
        """
        
        # 1. Get Input Path
        preprocess_path = self.ui.preprocessFilePath.text()
        if not preprocess_path:
            print("Error: No preprocessed file selected.")
            return    
        
        # 3. Get Save Path
        preprocess_path_obj = pathlib.Path(preprocess_path)
        output_file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model Training Data",
            "./models/"+preprocess_path_obj.name.replace(".pkl", "_model_data.pkl"),
            "Pickle Files (*.pkl)"
        )

        if not output_file_path:
            return
            
        self.ui.modelWritePath.setText(output_file_path)
        
        print("Preparing data and saving...")
        
        # Define architecture constants 
        NCOMP = 16 # Example default

        self.train_model_cvae_rf(
            run_pairs=[preprocess_path],
            n_comp=NCOMP,
            output_file_path=output_file_path
        )
        print("Model data written successfully.")
        self.setup_model_ui() # This allows the user to load the new model immediately.
            
    @Slot()
    def load_selected_model(self):
        """Loads the selected model and creates the worker immediately."""
        filename = self.ui.modelName.currentText()
        if not filename:
            return

        full_path = os.path.join(self.model_folder, filename)
        
        # 1. Clean up existing worker if it exists
        if self.worker is not None:
            self.handle_log("Cleaning up previous worker...")
            self.worker.stop()
            self.worker.new_prediction_signal.disconnect()
            self.worker.new_pv_values.disconnect()
            # Ensure the start button is reset
            self.ui.startPauseButton.setText("Start")

        self.current_model_path = full_path
        self.handle_log(f"Loading Model & Initializing Sync Buffer: {filename}")

        try:
            # 2. Instantiate worker immediately (this triggers load_model_resources)
            self.worker = InferenceWorker(self.current_model_path)
            # 3. Connect signals
            self.worker.new_prediction_signal.connect(self.update_image_display)
            self.worker.new_pv_values.connect(self.update_pv_values)
            # Re-enable this if you uncomment the emit in InferenceWorker
            # self.worker.new_log_signal.connect(self.handle_log)
            # 4. Populate the PV table
            self.setup_pv_table()
            
            self.handle_log("Worker ready. Press 'Start' to begin acquisition.")
        except Exception as e:
            self.handle_log(f"Failed to initialize worker: {e}")
            self.worker = None

    def update_pv_values(self, latest_inputs, bypass_flags):
        """
        Updates the 'Current Value' and 'Bypassed' columns of the PV table.
        
        Args:
            latest_inputs (np.ndarray): The raw values used for the latest inference (shape: N,)
            bypass_flags (list of bool): True if PV was bypassed/defaulted, False otherwise.
        """
        if not hasattr(self.ui, 'pvTable') or self.ui.pvTable.rowCount() == 0:
            return

        # Ensure we don't go out of bounds
        row_count = self.ui.pvTable.rowCount()
        
        # Disable sorting temporarily to improve performance during update
        self.ui.pvTable.setSortingEnabled(False)

        for row in range(row_count):
            if row < len(latest_inputs):
                # 1. Update Current Value
                val = latest_inputs[row]
                item_val = self.ui.pvTable.item(row, 3)
                if not item_val:
                    item_val = QTableWidgetItem()
                    self.ui.pvTable.setItem(row, 3, item_val)
                item_model_min_value = float(self.ui.pvTable.item(row, 1).text())
                item_model_max_value = float(self.ui.pvTable.item(row, 2).text())
                item_val.setText(f"{val:.4e}")
                if val>item_model_max_value:
                    item_val.setForeground(QColor("red"))
                elif val<item_model_min_value:
                    item_val.setForeground(QColor("red"))
                else:
                    item_val.setForeground(QColor("black"))

            if row < len(bypass_flags):
                # 2. Update Bypassed Status
                is_bypassed = bypass_flags[row]
                item_bypass = self.ui.pvTable.item(row, 4)
                if not item_bypass:
                    item_bypass = QTableWidgetItem()
                    self.ui.pvTable.setItem(row, 4, item_bypass)
                
                # Update text and background color for visibility
                if is_bypassed:
                    item_bypass.setText("YES")
                    item_bypass.setBackground(QColor("red"))
                    item_bypass.setForeground(QColor("white"))
                else:
                    item_bypass.setText("No")
                    item_bypass.setBackground(QColor("white")) # Or default
                    item_bypass.setForeground(QColor("black"))

        # Re-enable sorting (if you use it)
        self.ui.pvTable.setSortingEnabled(True)

    @Slot()
    def toggle_acquisition(self):
        """Toggles between Start and Pause states using the pre-loaded worker."""
        btn = self.ui.startPauseButton
        
        if self.worker is None:
            self.handle_log("Error: No model/worker loaded. Please click 'Load Model'.")
            return

        # Check if worker is running
        if self.worker.isRunning():
            # PAUSE
            self.worker.stop()
            btn.setText("Start")
            self.handle_log("Acquisition Paused")
        else:
            # START
            btn.setText("Pause")
            self.handle_log("Acquisition Started")
            self.worker.start()

    def setup_image_plot(self):
        """
        Embeds a pyqtgraph ImageView into the ui.imageContainer
        and configures it for a 'jet'-like heatmap.
        """
        # Create the ImageView widget
        # view=pg.PlotItem() allows axes/titles if you want them
        self.img_view = pg.ImageView(view=pg.PlotItem())
        
        # Hide the ROI button and Menu button for a cleaner look
        self.img_view.ui.roiBtn.hide()
        self.img_view.ui.menuBtn.hide()
        
        # Add it to the container layout
        # Assumes imageContainer has a layout (Vertical/Horizontal)
        if self.ui.imageContainer.layout() is None:
            from qtpy.QtWidgets import QVBoxLayout
            self.ui.imageContainer.setLayout(QVBoxLayout())
            
        self.ui.imageContainer.layout().addWidget(self.img_view)
        
        # --- SETTING THE COLORMAP (JET) ---
        # pyqtgraph has built-in colormaps. 
        # We assume data is 0..1 or similar; the histogram will auto-scale.
        
        # Define the positions (0.0 to 1.0) and colors (RGBA byte tuples) for 'jet'
        # Blue -> Cyan -> Yellow -> Red
        pos = np.array([0.0, 0.33, 0.66, 1.0])
        color = np.array([
            [0, 0, 128, 255],   # Dark Blue
            [0, 255, 255, 255], # Cyan
            [255, 255, 0, 255], # Yellow
            [128, 0, 0, 255]    # Dark Red
        ], dtype=np.ubyte)
        
        # Create the color map
        cmap = pg.ColorMap(pos, color)
        
        # Apply it
        self.img_view.setColorMap(cmap)

        # Create the PlotWidget widget
        self.current_view = pg.PlotWidget()
        
        # Add it to the container layout
        # Assumes imageContainer has a layout (Vertical/Horizontal)
        if self.ui.currentProfile.layout() is None:
            from qtpy.QtWidgets import QVBoxLayout
            self.ui.currentProfile.setLayout(QVBoxLayout())
            
        self.ui.currentProfile.layout().addWidget(self.current_view)
        
    
        self.current_view.enableAutoRange(axis = 'x', enable = True)
        self.current_view.enableAutoRange(axis = 'y', enable = True)
        self.current_view.setLabel('bottom', text='t', units='fs')
        self.current_view.setLabel('left', text='Current', units='A')
        self.current_view.showGrid(x=True, y=True)
        self.current_view.setMouseEnabled(x=False, y=False)

        # Create the PlotWidget widget
        self.energy_view = pg.PlotWidget()
        
        # Add it to the container layout
        # Assumes imageContainer has a layout (Vertical/Horizontal)
        if self.ui.energyProfile.layout() is None:
            from qtpy.QtWidgets import QVBoxLayout
            self.ui.energyProfile.setLayout(QVBoxLayout())
            
        self.ui.energyProfile.layout().addWidget(self.energy_view)
        
    
        self.energy_view.enableAutoRange(axis = 'x', enable = True)
        self.energy_view.enableAutoRange(axis = 'y', enable = True)
        self.energy_view.setLabel('bottom', text='Slice Charge', units='Arb.U.')
        self.energy_view.setLabel('left', text='Energy', units='Arb.U.')
        self.energy_view.showGrid(x=True, y=True)
        self.energy_view.setMouseEnabled(x=False, y=False)


    @Slot(np.ndarray)
    def update_image_display(self, image_data):
        """
        Receives the numpy array from the worker.
        Renders it into a Jet Heatmap using Matplotlib, and then updates the display.
        Updates horizontal and vertical projections.
        """
        # 1. Update Image (The FAST way)
        # autoRange=False prevents the view from resetting zoom every frame
        # autoLevels=False keeps the contrast stable (or True if you want auto-scaling)
        if hasattr(self, 'img_view'):
            # Note: pyqtgraph often expects (Cols, Rows) or Transposed data 
            # depending on how you want X/Y to align. 
            # If the image is rotated, try: image_data.T
            self.img_view.setImage(image_data.T, autoRange=False, autoLevels=False)
        # 2. Update Projections
        try:
            # image_data shape is (Rows, Cols) -> (Y, X)
            y_proj = np.sum(image_data, axis=1) # Sum across columns -> (Rows,)
            x_proj = np.sum(image_data, axis=0) # Sum across rows -> (Cols,)
            
            # Update Energy Spectrum (Vertical)
            y_indices = len(y_proj) - np.arange(len(y_proj)) # Flip due to orientation.
            self.energy_view.clear()
            # Plotting y_proj as X-values and indices as Y-values to make it "vertical"
            self.energy_view.plot(y_proj, y_indices)

            x_indices = np.arange(len(x_proj)) * self.worker.xtcalibrationfactor
            x_proj = x_proj / self.worker.xtcalibrationfactor * 1.602e-4 # 1.602e-19 [C]/1e-15 [s]

            self.current_view.clear()
            self.current_view.plot(x_indices, x_proj)
                
        except Exception as e:
            pass
    @Slot(int, int)
    def handle_table_click(self, row, col):
        """
        Handles clicks on the PV Table.
        If the 'Bypassed' column (col 4) is clicked:
        1. Pauses the worker immediately.
        2. Toggles the bypass flag for that PV in the worker.
        3. Updates the UI to reflect the change.
        """
        # Only react to clicks on the Bypassed column (Index 4)
        if col != 4:
            return

        # Ensure worker exists
        if not self.worker:
            return

        # 1. Pause the Worker immediately
        if self.worker.isRunning():
            self.handle_log(f"Worker paused to modify bypass for row {row}.")
            self.worker.stop() # This sets running=False and waits for the thread
            self.ui.startPauseButton.setText("Start")
        
        # 2. Toggle the Bypass Flag
        # Ensure we are within bounds
        if row < len(self.worker.is_pv_bypassed):
            current_state = self.worker.is_pv_bypassed[row]
            
            # Toggle (0 -> 1, 1 -> 0)
            new_state = 0 if current_state else 1
            self.worker.is_pv_bypassed[row] = new_state
            
            state_str = "Bypassed" if new_state else "Active"
            pv_name = self.worker.var_names[row]
            self.handle_log(f"PV '{pv_name}' set to {state_str}.")

            # 3. Update UI Immediately (Visual Feedback)
            item = self.ui.pvTable.item(row, 4)
            if not item:
                item = QTableWidgetItem()
                self.ui.pvTable.setItem(row, 4, item)
            
            if new_state:
                item.setText("YES")
                item.setBackground(QColor("red"))
                item.setForeground(QColor("white"))
            else:
                item.setText("No")
                item.setBackground(QColor("white"))
                item.setForeground(QColor("black"))
                
    def preprocess_and_save_lps(self, daq_file_path, x_roi, y_roi, xrange, yrange, output_file_path):
        """
        PreProcesses DAQ file with XTCAV phase classification, centroid correction, 
        and Center of Mass centering.
        
        Args:
            daq_file_path (str): Path to .mat file.
            x_roi (tuple): (min, max) for X cropping.
            y_roi (tuple): (min, max) for Y cropping.
            xrange (int): Aspect ratio/size param for X.
            yrange (int): Aspect ratio/size param for Y.
            output_file_path (str): Output pickle filename.
        """
        daq_path_obj = pathlib.Path(daq_file_path)
        # Define XTCAV calibration
        krf = 239.26
        cal = 1167 # um/deg  http://physics-elog.slac.stanford.edu/facetelog/show.jsp?dir=/2025/11/13.03&pos=2025-$
        streakFromGUI = cal*krf*180/np.pi*1e-6#um/um

        # Sets the main beam energy
        mainbeamE_eV = 10e9
        # Sets the dnom value for CHER
        dnom = 59.8e-3

        ## Below value MUST be specified for DAQs with unwanted refraction patterns, etc.
        roi_xrange = x_roi#(200, 1000)#For run 12710: (400, 700)
        roi_yrange = y_roi#(400, 600)#For run 12710: (400, 600), 12691: (0, 236)
        # Loads dataset
        dataloc = daq_path_obj
        directory_path = dataloc.parent
        mat = loadmat(dataloc,struct_as_record=False, squeeze_me=True)
        data_struct = mat['data_struct']

        # Extracts number of steps
        stepsAll = data_struct.params.stepsAll
        if stepsAll is None or len(np.atleast_1d(stepsAll)) == 0:
            stepsAll = [1]
        step_list = stepsAll # TODO: Read Step List from GUI

        # calculate xt calibration factor
        _xtcalibrationfactor = data_struct.metadata.DTOTR2.RESOLUTION*1e-6/streakFromGUI/3e8
        # gaussian filter parameter
        hotPixThreshold = 1e3
        sigma = 1
        threshold = 5
        print("Processing steps:", step_list)
        bsaScalarData, bsaVars = extractDAQBSAScalars(data_struct, step_list)
        bsaScalarData = apply_tcav_zeroing_filter(bsaScalarData, bsaVars)
        print("Extracted BSA scalar data shape:", bsaScalarData.shape)
        print("bhsVars:"+str(bsaVars))
        ampl_idx = next(i for i, var in enumerate(bsaVars) if 'TCAV_LI20_2400_A' in var)
        xtcavAmpl = bsaScalarData[ampl_idx, :]

        phase_idx = next(i for i, var in enumerate(bsaVars) if 'TCAV_LI20_2400_P' in var)
        xtcavPhase = bsaScalarData[phase_idx, :]

        xtcavOffShots = xtcavAmpl<0.1
        xtcavPhase[xtcavOffShots] = 0 #Set this for ease of plotting

        isChargePV = [bool(re.search(CHARGE_PV_U, pv)) for pv in bsaVars]
        pvidx = [i for i, val in enumerate(isChargePV) if val]
        charge = bsaScalarData[pvidx, :] * 1.6e-19  # in C 

        minus_90_idx = np.where((xtcavPhase >= -91) & (xtcavPhase <= -89))[0]
        plus_90_idx = np.where((xtcavPhase >= 89) & (xtcavPhase <= 91))[0]
        off_idx = np.where(xtcavPhase == 0)[0]
        all_idx = np.append(minus_90_idx,plus_90_idx)
        # Extract current profiles and 2D LPS images 
        xtcavImages_list = []
        xtcavImages_list_raw = []
        horz_proj_list = []
        LPSImage = [] 
        xtcavImages_centroid_uncorrected, xtcavImages_raw, horz_proj, LPSImage = extract_processed_images(data_struct, '', xrange, yrange, hotPixThreshold, sigma, threshold, step_list, roi_xrange, roi_yrange, do_load_raw=False, directory_path=directory_path)# do_load_raw = False by default.

        # Use step_list to filter step_data
        scalars_data = data_struct.scalars.common_index-1
        step_data = data_struct.scalars.steps[scalars_data]
        print("Processing steps:", step_list)
        step_data_filtered = np.isin(step_data, step_list)
        step_data_tmp = np.array(step_data[step_data_filtered])
        print(off_idx)
        print("Those should be the same lengths:")
        print(step_data_tmp.shape)
        print(xtcavImages_centroid_uncorrected.shape)
        xtcavImages, horz_proj, _, centroid_corrections = apply_centroid_correction(xtcavImages_centroid_uncorrected, off_idx, steps_if_stepwise=step_data_tmp, do_rotate=False)

         # Calculate target center (image center)
        cy, cx = xtcavImages.shape[0] / 2.0, xtcavImages.shape[1] / 2.0
        
        for i in range(xtcavImages.shape[2]):
            img = xtcavImages[:, :, i]
            
            # Skip empty images (e.g. from zeroing filter) to avoid errors
            if np.sum(img) == 0:
                continue
                
            # Calculate CoM
            com_y, com_x = center_of_mass(img)
            
            # Calculate shift required
            shift_y = cy - com_y
            shift_x = cx - com_x
            
            # Apply shift (using order=1 for linear interpolation or 0 for nearest)
            # We use shift from scipy.ndimage
            xtcavImages[:, :, i] = shift(img, shift=(shift_y, shift_x), mode='constant', cval=0.0)


        max = []
        #This is 1 based indexing from matlab!!!
        scalar_common_idx = []

        # At this point, xtcavImage, currentProfile, LPSImage are already filtered with image common index, hence they are dense from 0 to N-(some lost images).
        # bsaScalarData and xtcavPhase are filtered with scalar common index too, so xtcavImage[i] corresponds to bsaScalarData[i] and xtcavPhase[i].
        # However, we still have to store the scalar indices in the pickle if we want to map back to scalars in a different notebook.
        for ij in range(xtcavImages.shape[2]):
            scal_idx = data_struct.scalars.common_index[ij]
            # Corresponding scalar index
            scalar_common_idx.append(scal_idx)
            max.append(np.max(xtcavImages[:, :, ij]))
        max = np.array(max)
        scalar_common_idx = np.array(scalar_common_idx)


        #goodShots = np.arange(xtcavImages.shape[2])
        goodShots = np.where((max > 200))[0] # (np.abs(mu1 - mu2) > 20)  & (a_ratio > 0.05) & (a_ratio < 20)
        #goodShots_twobunch_tcav = np.where((R_squared > 0.97) & (amp1 < 50) & ((mu1 > mu2) & (amp1 < amp2)))[0]
        # --- Filter the data based on the indexing logic from the MLP setup cell ---

        xtcavImages_good = xtcavImages[:,:, goodShots]

        # ----------------------------------------------------------------

        # 1. Define the filename
        filename = output_file_path

        data_to_save = {
            'LPSImage': xtcavImages_good,
            #This is 1 based indexing from matlab!!!
            'scalarCommonIndex': scalar_common_idx[goodShots],
            'daqPath': daq_file_path,
            'xtcalibrationfactor': _xtcalibrationfactor,
            'description': '2D LPS Images (flattened) for good shots only, filtered by all_idx and then goodShots.'
        }

        # 2. Save the data using pickle
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Successfully saved good shot LPS Image data to '{filename}'.")
        print(f"Saved image shape: {xtcavImages_good.shape}")
            
    def train_model_cvae_rf(self, run_pairs, n_comp, output_file_path):
        # ----------------------------------------------------------------------
        # 2. Initialize lists for concatenation
        # ----------------------------------------------------------------------
        all_images = []
        all_predictors = []
        all_indices = []

        print("Starting multi-run data loading and concatenation...")

        # ----------------------------------------------------------------------
        # 3. Loop through runs, load data, and concatenate
        # ----------------------------------------------------------------------
        charge_merged = []
        for pickle_filename in run_pairs:
            
            # --- A. Load Processed LPSImage Data and Good Shots Index ---

            try:
                with open(pickle_filename, 'rb') as f:
                    data = pickle.load(f)
                
                LPSImage_good = data['LPSImage'] # Filtered LPS images
                # This 'goodShots' index is relative to the phase-filtered data (all_idx).
                goodShots_scal_common_index = data['scalarCommonIndex'] 
                
                print(f"Loaded pickle_filename: LPSImage shape {LPSImage_good.shape}")
                
            except FileNotFoundError:
                print(f"Skipping pickle_filename: Pickle file not found at {pickle_filename}")
                continue
            
            # --- B. Load and Filter Predictor Data (BSA Scalars) ---
            
            # 1. Load data_struct
            dataloc = data['daqPath']  # Assuming the path to the .mat file is stored in the pickle
            try:
                mat = loadmat(dataloc,struct_as_record=False, squeeze_me=True)
                data_struct = mat['data_struct']
            except FileNotFoundError:
                print(f"Skipping pickle_filename: .mat file not found at {dataloc}")
                continue

            # 2. Extract full BSA scalars (filtered by step_list if needed)
            # Don't filter by common index here, we'll do it with the goodShots scalar common index loaded from the file
            bsaScalarData, bsaVars = extractDAQBSAScalars(data_struct, filter_index=False)
            bsaScalarData = apply_tcav_zeroing_filter(bsaScalarData, bsaVars)

            # 3. Extract non BSA scalars the same way
            nonBsaScalarData, nonBsaVars = extractDAQNonBSAScalars(data_struct, filter_index=False, debug=False, s20=True)
            nonBsaScalarData = apply_tcav_zeroing_filter(nonBsaScalarData, nonBsaVars)

            # 4. Combine BSA and non-BSA scalar data
            bsaScalarData = np.vstack((bsaScalarData, nonBsaScalarData))
            allVars = bsaVars + nonBsaVars

            # 5. Filter BSA data using the final index
            # goodShots_scal_common_index is 1 based indexing from MATLAB, convert to 0 based
            bsaScalarData_filtered = bsaScalarData[:, goodShots_scal_common_index - 1]
            
            isChargePV = [bool(re.search(CHARGE_PV_U, pv)) for pv in bsaVars]
            if isChargePV:
                # Extract charge data
                pvidx = [i for i, val in enumerate(isChargePV) if val]
                charge = bsaScalarData[pvidx, :][0] * 1.6e-19  # in C 
                charge_filtered = charge[goodShots_scal_common_index - 1]
            # 6. Construct the predictor array
            predictor_current = np.vstack(bsaScalarData_filtered).T
            
            # C. Append to master lists
            all_images.append(LPSImage_good)
            all_predictors.append(predictor_current)
            charge_merged.append(charge_filtered)
            
        # ----------------------------------------------------------------------
        # 4. Concatenate and finalize arrays
        # ----------------------------------------------------------------------


        # Combine all data arrays from the runs
        images_tmp = np.concatenate(all_images, axis=0)
        print(f"TMP Shape: {images_tmp.shape}")
        # Set image half dimensions (should match preprocessing)
        yrange = images_tmp.shape[0]//2
        xrange = images_tmp.shape[1]//2
        images_tmp = images_tmp.transpose(2, 0, 1)
        predictor_tmp = np.concatenate(all_predictors, axis=0)
        charge = np.concatenate(charge_merged, axis=0)


        print("\n--- Final Concatenated Data Shapes ---")
        print(f"Total LPS Images (images): {images_tmp.shape}")
        print(f"Total Predictors (predictor): {predictor_tmp.shape}")
        self.update_image_display(np.average(images_tmp, axis = 0))

        ampl_idx = next(i for i, var in enumerate(bsaVars) if 'TCAV_LI20_2400_A' in var)
        xtcavAmpl = predictor_tmp[:, ampl_idx]

        phase_idx = next(i for i, var in enumerate(bsaVars) if 'TCAV_LI20_2400_P' in var)
        xtcavPhase = predictor_tmp[:, phase_idx]
        xtcavOffShots = xtcavAmpl<0.1
        xtcavPhase[xtcavOffShots] = 0 #Set this for ease of plotting
        # 
        near_minus_90_idx = np.where((xtcavPhase >= -90.55) & (xtcavPhase <= -89.55))[0]
        near_plus_90_idx = np.where((xtcavPhase >= 89.55) & (xtcavPhase <= 90.55))[0]
        lps_idx = near_minus_90_idx.tolist() + near_plus_90_idx.tolist()
        # Flip image horizontally for -90 deg phase
        images_flipped = images_tmp.copy()
        images_flipped[near_minus_90_idx, :, :] = np.flip(images_tmp[near_minus_90_idx, :, :], axis=2)

        from Python_Functions.functions import exclude_bsa_vars
        excluded_var_idx = exclude_bsa_vars(allVars)
        bsaVarNames_cleaned = [var for i, var in enumerate(allVars) if i not in excluded_var_idx]
        predictor_tmp_cleaned = np.delete(predictor_tmp, excluded_var_idx, axis=1)[lps_idx, :]
        print(f"Predictor shape after excluding variables: {predictor_tmp_cleaned.shape}")

        LPSimg_prezoom = images_flipped[lps_idx]
        charge_filtered = charge[lps_idx]
        print(f"LPS Image shape after filtering: {LPSimg_prezoom.shape}")


        # Calibration
        # Define XTCAV calibration
        krf = 239.26
        cal = 1167 # um/deg  http://physics-elog.slac.stanford.edu/facetelog/show.jsp?dir=/2025/11/13.03&pos=2025-$
        streakFromGUI = cal*krf*180/np.pi*1e-6#um/um
        _xtcalibrationfactor = data_struct.metadata.DTOTR2.RESOLUTION*1e-6/streakFromGUI/3e8
        # Flags
        # If enabled, GMM fit is weighted to predict better current profile rather than overall image fit
        do_current_profile = True

        # Interpolate LPS images to square pixels for CVAE training.
        # 2* yrange, 2* xrange to 200x200
        # CVAE model assumes square images with 200x200 pixels, forced by convolutional layers
        from scipy.ndimage import zoom
        LPSimg_resized = np.zeros((LPSimg_prezoom.shape[0], 200, 200), dtype=LPSimg_prezoom.dtype)
        for i in range(LPSimg_prezoom.shape[0]):
            LPSimg_resized[i] = zoom(LPSimg_prezoom[i], (200/(2*yrange), 200/(2*xrange)), order=1)
        LPSimg = LPSimg_resized

        
        # Number of components for the CVAE.
        # Too low, the model will not capture the complexity of the data.
        # Too high, the prediction task becomes too difficult.
        NCOMP = n_comp

        INPUT_CHANNELS = 1 # Assuming LPSimg is grayscale/single-channel data
        # 1. Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        # 2. Initialize Model and Optimizer
        model_cvae = CVAE(latent_dim=NCOMP).to(device)
        optimizer = torch.optim.Adam(model_cvae.parameters(), lr=1e-3)

        # LPSimg is a numpy array of shape [n_samples, 200, 200].
        BATCH_SIZE = 16
        # ---------------------------------------------------

        # Convert numpy data to PyTorch Tensor, add the Channel dimension (C=1)
        LPSimg_tensor = torch.from_numpy(LPSimg).unsqueeze(1).float()
        LPSimg_tensor /= LPSimg_tensor.max()
        # Create a simple DataLoader for the data
        from torch.utils.data import TensorDataset, DataLoader

        from Python_Functions.cvae import proj_vae_loss
        dataset = TensorDataset(LPSimg_tensor)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 4. Training Loop
        N_EPOCHS = 8
        print("\nStarting training loop ({} epochs)...".format(N_EPOCHS))

        for epoch in range(1, N_EPOCHS + 1):
            total_loss = 0
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.to(device)
                
                # Forward pass
                reconstruction, mu, logvar = model_cvae(data)
                # reconstruction /= reconstruction.max()
                data = data.clamp(min = 0)
                # Calculate loss
                # Like normal loss, vae_loss is not great for predicting LPS with small isolated features.
                # Using projection loss to improve reconstruction of small features, which are enhanced logarithmically in the projection.
                loss = proj_vae_loss(reconstruction, data, mu, logvar, strength=0.5)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            avg_loss = total_loss / len(dataset)
            print(f"Epoch {epoch}/{N_EPOCHS}, Average VAE Loss: {avg_loss:.4f}")

        # Encode all LPS images to latent z parameters.
        latent_z_array = np.zeros((LPSimg.shape[0], NCOMP )) 
        for i in range(LPSimg.shape[0]):
            mu_tensor = model_cvae.generate_latent_mu(torch.from_numpy(LPSimg[i]/LPSimg[i].max()).unsqueeze(0).unsqueeze(0).float().to(device))
            latent_z_array[i] = mu_tensor.cpu().detach().numpy()

        # Random Forest Regression to predict each latent parameter from the predictor variables

        # All rows are valid in this mock example
        valid_rows = [True for i in range(latent_z_array.shape[0])]
        predictor_filtered = predictor_tmp_cleaned[valid_rows]
        biGaussian_params_array_filtered = latent_z_array[valid_rows]
        print(f"After removing invalid rows, dataset shape: Predictors {predictor_filtered.shape}, Bi-Gaussian Params {biGaussian_params_array_filtered.shape}")

        # --- Original scaling and splitting logic follows ---

        x_scaler = MinMaxScaler()
        iz_scaler = MinMaxScaler()
        x_scaled = x_scaler.fit_transform(predictor_filtered)
        Iz_scaled = iz_scaler.fit_transform(biGaussian_params_array_filtered)



        # 80/20 train-test split
        x_train_full, x_test_scaled, Iz_train_full, Iz_test_scaled, ntrain, ntest = train_test_split(
            x_scaled, Iz_scaled, np.arange(Iz_scaled.shape[0]), test_size=0.2, random_state = 42)

        # 20% validation split 
        x_train_scaled, x_validation, Iz_train_scaled, y_validation = train_test_split(
            x_train_full, Iz_train_full, test_size=0.2, random_state = 42)

        # Convert to PyTorch tensors
        X_train = torch.tensor(x_train_scaled, dtype=torch.float32)
        x_validation = torch.tensor(x_validation, dtype=torch.float32)
        X_test = torch.tensor(x_test_scaled, dtype=torch.float32)
        Y_train = torch.tensor(Iz_train_scaled, dtype=torch.float32)
        y_validation = torch.tensor(y_validation, dtype=torch.float32)
        Y_test = torch.tensor(Iz_test_scaled, dtype=torch.float32)

        train_ds = TensorDataset(X_train, Y_train)
        train_dl = DataLoader(train_ds, batch_size=24, shuffle=True)

        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {Y_train.shape}")

        # --- 2. Model and Hyperparameter Setup ---
        # The Random Forest is initialized with its structure (number of trees, depth, etc.)
        # This replaces the PyTorch MLP class definition.

        print("\n--- Initializing Model ---")
        # Equivalent to: model = MLP(X_train.shape[1], Y_train.shape[1])
        # n_estimators is equivalent to the overall model complexity/capacity
        # max_depth controls the depth, similar to the number of layers/nodes.
        model = RandomForestRegressor(
            n_estimators=500,        # Number of trees (like epochs/steps, higher = more complex)
            max_depth=15,            # Max depth of trees (limits complexity)
            min_samples_leaf=5,      # Regularization/pruning parameter
            random_state=42,
            n_jobs=-1                # Use all available cores
        )

        # Note: In Random Forest, there is no separate 'optimizer' or 'learning rate'
        # as the training is done via deterministic tree growing (not gradient descent).


        # --- 4. Training and Evaluation (Equivalent to the Training Loop) ---
        # Random Forest is trained in a single 'fit' call, not in epochs.
        # We mimic the training block structure and calculate loss/metrics.

        t0 = time.time()
        print("\n--- Starting Model Fitting (One Shot) ---")

        # Fit the model (This is the entire 'training loop' for RF)
        model.fit(X_train, Y_train)

        t1 = time.time()

        # --- Evaluation ---

        # 1. Training Set Evaluation
        Y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(Y_train, Y_train_pred)

        # 2. Validation Set Evaluation
        Y_val_pred = model.predict(x_validation)
        val_mse = mean_squared_error(y_validation, Y_val_pred)


        print("\n--- Training Results ---")
        print(f"Total Fitting Time: {t1 - t0:.2f} seconds")
        print(f"Final Train MSE: {train_mse:.6f}")
        print(f"Final Validation MSE: {val_mse:.6f}")

        # To see the importance of the input features:
        print("\n--- Feature Importance ---")
        for i, importance in enumerate(model.feature_importances_):
            print(f"Feature {i} importance: {importance:.4f}")
        # Evaluate model
        pred_train_scaled = model.predict(X_train)
        pred_test_scaled = model.predict(X_test)

        # Inverse transform predictions
        pred_train_full = iz_scaler.inverse_transform(pred_train_scaled)
        pred_test_full = iz_scaler.inverse_transform(pred_test_scaled)
        Iz_train_true = iz_scaler.inverse_transform(Iz_train_scaled)
        Iz_test_true = iz_scaler.inverse_transform(Iz_test_scaled)
        elapsed = time.time() - t0
        print("Elapsed time [mins] = {:.1f} ".format(elapsed/60))

        # Compute R² score
        def r2_score(true, pred):
            RSS = np.sum((true - pred)**2)
            TSS = np.sum((true - np.mean(true))**2)
            return 1 - RSS / TSS if TSS != 0 else 0

        # Compute R² on scaled data, instead of the actual bi-Gaussian parameters, to avoid distortion from different scales
        print("Train R²: {:.2f} %".format(r2_score(Iz_train_scaled.ravel(), pred_train_scaled.ravel()) * 100))
        print("Test R²: {:.2f} %".format(r2_score(Iz_test_scaled.ravel(), pred_test_scaled.ravel()) * 100))

        time_stamp = time.strftime("%Y%m%d_%H%M%S")

        data_to_save = {
            'varNames': bsaVarNames_cleaned,
            'bsaVarNames': bsaVars,
            'nonBsaVarNames': nonBsaVars,
            'model': model,
            'iz_scaler': iz_scaler,
            'x_scaler': x_scaler,
            'xtcalibrationfactor': _xtcalibrationfactor,
            'image_model' : model_cvae,
            'architecture': {
                'ncomp': NCOMP,
                'xrange': xrange,
                'yrange': yrange,
                'type': 'CVAE Random Forest v2025.12'
            },
        }
        data_file = output_file_path
        with open(data_file, 'wb') as f:
            pickle.dump(data_to_save, f)
            
    @Slot(str)
    def handle_log(self, message):
        """Appends log messages to the PyDMLogDisplay"""
        if hasattr(self.ui, 'PyDMLogDisplay'):
            # PyDMLogDisplay might be a simple widget wrapping a TextEdit. 
            # If it's a standard QWidget as per UI file, we might need to find the child text edit
            # or assuming it has a method to add text. 
            # If it acts like a PyDM text display, we assume standard Qt methods work if exposed.
            
            # Attempt to find a method to append text
            if hasattr(self.ui.PyDMLogDisplay, 'appendPlainText'):
                self.ui.PyDMLogDisplay.appendPlainText(message)
            elif hasattr(self.ui.PyDMLogDisplay, 'append'):
                self.ui.PyDMLogDisplay.append(message)
            elif hasattr(self.ui.PyDMLogDisplay, 'setText'):
                # Append manually if only setText exists
                current = self.ui.PyDMLogDisplay.text() if hasattr(self.ui.PyDMLogDisplay, 'text') else ""
                self.ui.PyDMLogDisplay.setText(current + "\n" + message)
            else:
                # Fallback: Print to console if widget doesn't support text
                print(message)

    def closeEvent(self, event):
        # Clean up thread on close
        if self.worker:
            self.worker.stop()
        super().closeEvent(event)

    

# Entry point for PyDM
intelclass = VTCAVDisplay

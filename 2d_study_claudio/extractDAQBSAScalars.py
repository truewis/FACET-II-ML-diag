import numpy as np

def extractDAQBSAScalars(data_struct):
    """
    Extracts BSA scalar data and PV names from a DAQ data structure.

    Args:
        data_struct (dict): A mat_struct created from a .mat file
                            containing a 'scalars' field.

    Returns:
        tuple: (bsaScalarData, bsaVarPVs) where:
               - bsaScalarData (np.ndarray): Concatenated BSA scalar data.
               - bsaVarPVs (list): Concatenated list of BSA variable PV names.
    """

    dataScalars = data_struct.scalars

    # Assuming 'common_index' holds the indices for selecting data points.
    # In Python, indices are 0-based and the selection method depends on 
    # the exact data type, but typically involves slicing or fancy indexing.
    # MATLAB uses 1-based indexing, so we might need to adjust:
    # MATLAB's idx selects elements, e.g., varData(idx).
    # Python equivalent: varData[idx-1] if idx is 1-based, or varData[idx] if idx is 0-based boolean mask or integer array.
    # We'll assume 'idx' is a 0-based NumPy array of indices. If it's a 1-based MATLAB index, subtract 1.
    idx = dataScalars.common_index
    if idx is None:
        raise ValueError("data_struct['scalars'] must contain a 'common_index' key.")
        
    minVal = np.min((np.array(idx)))
    # Check if idx is 1-based (min(idx) >= 1) and convert to 0-based if necessary.
    # This is a common conversion for data migrated from MATLAB.
    if minVal >= 1:
        idx = idx - 1 

    # Find the names of all BSA lists (keys starting with 'BSA_List_')
    fNames = list(dataScalars._fieldnames)
    bsaListNames = [name for name in fNames if name.startswith('BSA_List_')]
    
    # isBSA will hold the actual names of the BSA lists
    isBSA = bsaListNames
    
    bsaScalarData = []  # Will store NumPy arrays
    bsaVarPVs = []      # Will store lists of strings (PV names)

    # Loop through all BSA lists and concatenate data
    for listName in isBSA:
        bsaList = getattr(dataScalars, listName)  # Go through each BSA list
        
        # Extract the BSA variable names (keys in the BSA list dictionary)
        varNames = list(bsaList._fieldnames)
        
        bsaListData_rows = []
        current_list_PVs = [] # Store PV names for this specific list
        
        # Loop through all variables in the current BSA list
        for varName in varNames:
            varData = getattr(bsaList, varName)
            
            # MATLAB's isempty(varData) check:
            if varData is None or (isinstance(varData, (np.ndarray, list)) and len(varData) == 0):
                # Ignore variables with empty data
                continue
            else:
                # 1. Select data using the common index (idx)
                # Ensure varData is a NumPy array for consistent indexing
                varData = np.array(varData)
                # Select the indexed data points
                varData_selected = varData[idx]
                
                # 2. MATLAB's varData(isnan(varData))=0 equivalent
                # Replace NaN values with 0
                if np.issubdtype(varData_selected.dtype, np.number):
                    varData_selected[np.isnan(varData_selected)] = 0

                # 3. Store the processed row data
                # MATLAB stores it as row vector: bsaListData(nVar,:) = varData
                bsaListData_rows.append(varData_selected)
                
                # 4. Store the PV name (only for non-empty variables)
                current_list_PVs.append(varName)
        
        # Concatenate PV names (already filtered for non-empty)
        bsaVarPVs.extend(current_list_PVs)
        
        # Concatenate the scalar data for the current list
        if bsaListData_rows:
            # Stack the rows vertically (MATLAB's implicit row assignment/concatenation)
            bsaListData = np.vstack(bsaListData_rows)
            # MATLAB's cat(1, a, b) concatenates vertically (rows)
            if len(bsaScalarData)==0:
                bsaScalarData = bsaListData
            else:
                bsaScalarData = np.concatenate((bsaScalarData, bsaListData), axis=0)
    
    # Final output conversion to NumPy array if it's still an empty list
    if bsaScalarData.size==0:
        bsaScalarData = np.empty((0, 0))

    # bsaVarPVs is already a list of strings
    return bsaScalarData, bsaVarPVs

# Example usage (assuming some mock data structure):
# data_struct = {
#     'scalars': {
#         'common_index': np.array([0, 1, 2, 5]),  # 0-based indices to select
#         # If it were 1-based [1, 2, 3, 6], the code would convert it.
#         'BSA_List_1': {
#             'PV_A': np.array([10, 20, 30, 40, 50, 60]),
#             'PV_B': np.array([11, 21, np.nan, 41, 51, 61]),
#             'PV_C': [] # Empty variable
#         },
#         'BSA_List_2': {
#             'PV_X': np.array([100, 200, 300, 400, 500, 600]),
#             'PV_Y': None # Empty variable
#         }
#     }
# }

# bsaData, bsaPVs = extractDAQBSAScalars(data_struct)
# print("BSA Scalar Data (NumPy Array):\n", bsaData)
# print("\nBSA Variable PVs (List):\n", bsaPVs)
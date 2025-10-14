import numpy as np

def extractDAQnonBSAScalars(data_struct):
    """
    Extracts non-BSA scalar data and PV names from a DAQ data structure.

    Args:
        data_struct (mat_struct): A mat_struct object containing a 'scalars' field
                                  loaded from a .mat file (e.g., using loadmat(..., struct_as_record=False)).

    Returns:
        tuple: (nonbsaScalarData, nonbsaVarPVs) where:
               - nonbsaScalarData (np.ndarray): Concatenated non-BSA scalar data (N_vars x N_shots).
               - nonbsaVarPVs (list): Concatenated list of non-BSA variable PV names.
    """

    # --- 1. Initial Setup and Index Conversion ---
    
    # MATLAB: dataScalars = data_struct.scalars;
    dataScalars = data_struct.scalars 
    
    # MATLAB: idx = dataScalars.common_index;
    idx = dataScalars.common_index
    if idx is None:
        raise ValueError("data_struct.scalars must contain a 'common_index' attribute.")
        
    # Check if idx is 1-based (MATLAB default) and convert to 0-based if necessary
    minVal = np.min(np.array(idx))
    if minVal >= 1:
        idx = np.array(idx) - 1  # Convert to 0-based array for Python indexing

    # --- 2. Find non-BSA List Names ---
    
    # MATLAB: fNames = fieldnames(dataScalars);
    # MATLAB: nonbsaListNames = regexp(fNames,'nonBSA_List_');
    # MATLAB: isnonBSA = find(nonbsaLists);
    
    # The convention in Python for mat_struct is using the private attribute _fieldnames.
    fNames = list(dataScalars._fieldnames)
    
    # Identify fields that start with 'nonBSA_List_'
    nonbsaListNames = [name for name in fNames if name.startswith('nonBSA_List_')]
    
    # --- 3. Initialize Output Arrays ---
    
    nonbsaScalarData = []  # Will store NumPy arrays (to be concatenated)
    nonbsaVarPVs = []      # Will store lists of strings (PV names)

    # --- 4. Loop through non-BSA Lists and Variables ---
    
    # MATLAB: for nList = 1:length(isnonBSA) ... nonbsaList = dataScalars.(fNames{listNum});
    for listName in nonbsaListNames:
        # Access the non-BSA list using getattr()
        nonbsaList = getattr(dataScalars, listName) 
        
        # MATLAB: varNames = fieldnames(nonbsaList);
        # Extract the variable names within the current list
        varNames = list(nonbsaList._fieldnames)
        
        nonbsaListData_rows = []
        current_list_PVs = [] 
        
        # Loop through all variables in the current non-BSA list
        # MATLAB: for nVar = 1:length(varNames) ... varData = nonbsaList.(varNames{nVar});
        for varName in varNames:
            varData = getattr(nonbsaList, varName)
            
            # MATLAB's isempty(varData) check:
            if varData is None or (isinstance(varData, (np.ndarray, list)) and len(varData) == 0):
                continue
            else:
                # 1. Select data using the common index (idx)
                # MATLAB: varData = varData(idx);
                varData = np.array(varData)
                varData_selected = varData[idx]
                
                # 2. Replace NaN values with 0
                # MATLAB: varData(isnan(varData))=0;
                if np.issubdtype(varData_selected.dtype, np.number):
                    varData_selected[np.isnan(varData_selected)] = 0

                # 3. Store the processed row data
                # MATLAB: nonbsaListData(nVar,:) = varData;
                nonbsaListData_rows.append(varData_selected)
                
                # 4. Store the PV name
                current_list_PVs.append(varName)
        
        # --- 5. Concatenate Data and PVs for the current list ---
        
        # MATLAB: nonbsaVarPVs = cat(1,nonbsaVarPVs,varNames);
        nonbsaVarPVs.extend(current_list_PVs)
        
        if nonbsaListData_rows:
            # Stack the rows vertically
            nonbsaListData = np.vstack(nonbsaListData_rows)
            
            # MATLAB: nonbsaScalarData = cat(1,nonbsaScalarData,nonbsaListData);
            if len(nonbsaScalarData) == 0:
                nonbsaScalarData = nonbsaListData
            else:
                # Concatenate vertically (axis=0)
                nonbsaScalarData = np.concatenate((nonbsaScalarData, nonbsaListData), axis=0)
    
    # Final output conversion to NumPy array if it's still an empty list
    if len(nonbsaScalarData) == 0:
         # Return a 0x0 empty array, matching an uninitialized MATLAB array's behavior
        nonbsaScalarData = np.empty((0, 0), dtype=np.float64) 

    return nonbsaScalarData, nonbsaVarPVs
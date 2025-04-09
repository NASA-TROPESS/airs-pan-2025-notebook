from copy import deepcopy
import numpy as np
import scipy.interpolate as spi


def column_integrate(
        VMRIn,
        airDensityIn,
        altitudeIn,
        minIndex=0,
        maxIndex=0,
        linearFlag=False,
        pressure=np.empty([0]),
        minPressure=np.empty([0]),
        maxPressure=np.empty([0])):

    function_name = "column_integrate: "

    # Because some of the arrays coming in are of shape [x,1], we reshape it to [x]
    if (VMRIn.shape == 2 and VMRIn.shape[1] == 1):
        VMRIn = np.reshape(VMRIn, (VMRIn.shape[0]))

    if (airDensityIn.shape == 2 and airDensityIn.shape[1] == 1):
        airDensityIn = np.reshape(airDensityIn, (airDensityIn.shape[0]))

    if (altitudeIn.shape == 2 and altitudeIn.shape[1] == 1):
        altitudeIn = np.reshape(altitudeIn, (altitudeIn.shape[0]))

    # AT_LINE 371 TOOLS/atmosphere.pro column_integrate
    airDensity = airDensityIn

    if np.amax(airDensity) > 1e25:
        # units passed in as molec/m3... should be passed in as molecules/cm3
        airDensity = airDensity / 1e6

    VMR = deepcopy(VMRIn)
    altitude = altitudeIn * 100 # centimeters!
    altitudeNew = altitude.copy()

    # screen by pressure.  For this we interpolate to the exact
    # pressures and then use indices to select
    # AT_LINE 387 TOOLS/atmosphere.pro column_integrate

    if len(minPressure) > 0 or len(maxPressure) > 0:
        altitudeNew = altitude
        pressureNew = pressure.copy()
        if len(maxPressure) > 0:
            altitudeNew0 = _interpol_1d(altitude, np.log(pressure), np.log(maxPressure))
            if np.amin(np.abs(altitudeNew0-altitudeNew)) > 0.0001:
                altitudeNew = np.concatenate((altitudeNew0, altitudeNew), axis=0)
                altitudeNew = np.sort(altitudeNew)
                pressureNew = np.concatenate((maxPressure, pressureNew), axis=0)
                pressureNew = np.sort(pressureNew)
                pressureNew = pressureNew[::-1]
            altmax = np.mean(altitudeNew0)

        if len(minPressure) > 0:
            altitudeNew0 = _interpol_1d(altitude, np.log(pressure), np.log(minPressure))
            if np.amin(np.abs(altitudeNew0-altitudeNew)) > 0.0001:
                altitudeNew = np.concatenate((altitudeNew0, altitudeNew), axis=0)
                altitudeNew = np.sort(altitudeNew)
                pressureNew = np.concatenate((minPressure, pressureNew), axis=0)
                pressureNew = np.sort(pressureNew)
                pressureNew = pressureNew[::-1]
            altmin = np.mean(altitudeNew0)

    if len(altitudeNew) != len(altitude):
        # here make maps, call this function with additional level
        # then map back to original levels.  Tested with new pressure 0.0005 hPa
        # away, and checked quantities are very small.

        my_levels = []
        for alt in altitude:
            ind = np.where(np.abs(alt - altitudeNew) < 0.001)[0]
            my_levels.append(ind[0])

        # linear in altitude
        map_new = _make_maps(altitudeNew, np.array(my_levels) + 1, i_linearFlag=True)

        airDensityNew2 = map_new['toState'].T @ airDensity

        if linearFlag:
            vmrNew2 = map_new['toState'].T @ VMR
        else:
            vmrNew2 = np.exp(map_new['toState'].T @ np.log(VMR))

        #airDensity = airDensityNew2
        #altitude = altitudeNew
        #VMR = vmrNew2

        maxIndexNew = np.argmin(np.abs(pressureNew - minPressure))
        minIndexNew = np.argmin(np.abs(pressureNew - maxPressure))

        # call this function again with additional levels...
        # use minIndex and maxIndex
        result = column_integrate(
            vmrNew2,
            airDensityNew2,
            altitudeNew/100, # need to update this as mult * 100 at start
            minIndex=minIndexNew,
            maxIndex=maxIndexNew,
            linearFlag=linearFlag,
            pressure=pressureNew
        )

        # map these back to original levels

        # layer quantities:  basically, remove one level at start and end
        # the partial layer is absorbed in the full level
        columnLayer = result.columnLayer[1:-1]
        # I do not know why but this does not work (and it should): columnLayer = vertical_rebin(result.columnLayer, pressureNew, pressure)
        level_to_layer = (result.level_to_layer.T @ map_new['toState'].T).T
        columnTotal = result.column
        derivative = map_new['toState'] @ result.derivative
        columnAirLayer = result.columnAirLayer[1:-1]
        columnAirTotal = result.columnAir
        derivativeLayer = np.matmul(derivative, level_to_layer)

    else:

        # no added levels

        # get indices
        if len(maxPressure) > 0:
            minIndex = np.argmin(np.abs(altmax - altitude))

        if len(minPressure) > 0:
            maxIndex = np.argmin(np.abs(altmin - altitude))
        # end if len(minPressure) > 0 or len(maxPressure) > 0:
        # AT_LINE 431 TOOLS/atmosphere.pro column_integrate

        if maxIndex == 0:
            maxIndex = len(altitude) - 1

        if maxIndex > len(altitude)-1:
            print(function_name, 'ERROR: maxIndex must be less than # altitude elements')
            assert False

        columnAirTotal = 0
        columnTotal = 0
        columnLayer = np.zeros(shape=(len(altitude)-1), dtype=np.float32)
        columnAirLayer = np.zeros(shape=(len(altitude)-1), dtype=np.float32)
        vmrLayer = np.zeros(shape=(len(altitude)-1), dtype=np.float32)

        # AT_LINE 445 TOOLS/atmosphere.pro column_integrate
        for jj in range(minIndex+1, maxIndex+1):
            x1 = airDensity[jj-1] * VMR[jj-1]
            x2 = airDensity[jj] * VMR[jj]
            dz = altitude[jj] - altitude[jj - 1]
            if x1 == x2:
                x1 = x2 * 1.0001

            # column for species
            columnLayer[jj - 1] = dz / np.log(np.abs(x1 / x2)) * (x1-x2)

            # column for air
            x1d = airDensity[jj-1]
            x2d = airDensity[jj]
            columnAirLayer[jj-1] = dz / np.log(np.abs(x1d / x2d)) * (x1d-x2d)

            if linearFlag:
                HV = (VMR[jj] - VMR[jj-1]) / dz
                HP = (np.log(x2d) - np.log(x1d)) / dz

                # override log calculations
                columnLayer[jj-1] = (x2-x1) / HP - HV * (x2d-x1d) / HP / HP

            columnAirTotal = columnAirTotal + columnAirLayer[jj-1]
            columnTotal = columnTotal + columnLayer[jj-1]
            vmrLayer[jj-1] = columnLayer[jj-1] / columnAirLayer[jj-1]

            # AT_LINE 475 TOOLS/atmosphere.pro column_integrate
            if np.isfinite(columnLayer[jj-1]) is False:
                print(function_name, 'Error... NaN')
                assert False
        # end for jj in range(minIndex+1, maxIndex+1):
        # AT_LINE 483 TOOLS/atmosphere.pro column_integrate

        # AT_LINE 486 TOOLS/atmosphere.pro column_integrate
        # DERIVATIVE of column total vs. VMR.  This will be used in error analysis
        # air density in molec / cm3
        # altitude in cm

        n = len(VMR)
        derivative = np.zeros(shape=(n), dtype=np.float64) # derivative of dcolumn/dVMR
        level_to_layer = np.zeros(shape=(n, n-1), dtype=np.float64) # map from levels to layers


        # S. Kulawik add 4/2024
        # but after discussions with Josh, sub-column should ALWAYS
        # be log/log to match OSS.  But that would be the flag passed in.
        # if the user truly wants linear columns, the pass in this flag and so 
        # make derivative consistent with layer quantities.
        if linearFlag == 1:
            # ssk 9/2024, should only calculate derivative for levels that impact
            # the range requested
            for jj in range(minIndex+1, maxIndex+1):
                x1 = airDensity[jj-1] * VMR[jj-1]
                x2 = airDensity[jj] * VMR[jj]
                x1d = airDensity[jj-1]
                x2d = airDensity[jj]
                dz = altitude[jj] - altitude[jj - 1]
                if x1 == x2:
                    x1 = x2 * 1.0001

                HV = (VMR[jj] - VMR[jj-1]) / dz
                HP = (np.log(x2d) - np.log(x1d)) / dz

                # @ depends on VMR[jj]; & depends on VMR[jj-1]
                #columnLayer[jj-1] = (x2@-x1&) / HP - HV@& * (x2d-x1d) / HP / HP

                derivative[jj] = derivative[jj] + airDensity[jj] / HP - 1 * (x2d-x1d) / dz / HP / HP 
                derivative[jj-1] = derivative[jj-1] - airDensity[jj-1] / HP + 1 * (x2d-x1d) / dz / HP / HP 

                ####
                #columnLayer[jj-1] = (airDensity[jj] * VMR[jj] - airDensity[jj-1] * VMR[jj-1]) / HP
                #  - (VMR[jj] - VMR[jj-1]) * (x2d-x1d) / (HP * HP * dz)
                #columnAirLayer[jj-1] = dz / np.log(np.abs(x1d / x2d)) * (x1d-x2d)
                #vmr_layer[jj-1] = (airDensity[jj] / HP - (x2d-x1d) / HP / HP / dz) / columnAirLayer[jj-1] * VMR[jj] +
                #                  (-airDensity[jj-1] / HP + (x2d-x1d) / HP / HP / dz) / columnAirLayer[jj-1] * VMR[jj-1]

                level_to_layer[jj, jj-1] = level_to_layer[jj, jj-1] + (airDensity[jj] / HP - (x2d-x1d) / HP / HP / dz) / columnAirLayer[jj-1]
                level_to_layer[jj-1, jj-1] = level_to_layer[jj-1, jj-1] + (-airDensity[jj-1] / HP + (x2d-x1d) / HP / HP / dz) / columnAirLayer[jj-1]

        else: 
            # AT_LINE 495 TOOLS/atmosphere.pro column_integrate
            # ssk 9/2024, should only calculate derivative for levels that impact
            # the answer
            for jj in range(minIndex+1, maxIndex+1):
                x1 = airDensity[jj-1] * VMR[jj-1]
                x2 = airDensity[jj] * VMR[jj]
                dz = altitude[jj] - altitude[jj-1]

                term1 = np.log(np.abs(x1/x2))
                term2 = x1 - x2
                derivative[jj] = derivative[jj] - dz / term1 * airDensity[jj] + dz / term1 / term1 * term2 / VMR[jj]
                derivative[jj-1] = derivative[jj-1] + dz / term1 * airDensity[jj-1] - dz / term1 / term1 * term2 / VMR[jj-1]

                # since above is d(column gas)/dVMR, change to 
                # d(layer # VMR) / d(levelVMR) by dividing by the air density for this layer.
                # air density for this layer: take above column measurements where VMR = 1.  
                # The dz cancels
                x1d = airDensity[jj-1]
                x2d = airDensity[jj]
                factor = np.log(np.abs(x1d/x2d))/(x1d-x2d)

                level_to_layer[jj, jj-1] = level_to_layer[jj, jj-1] + (1/term1/term1*term2/VMR[jj] - 1/term1*airDensity[jj])*factor
                level_to_layer[jj-1, jj-1] = level_to_layer[jj-1, jj-1] + (1/term1*airDensity[jj-1] - 1/term1/term1*term2/VMR[jj-1])*factor
            # end for jj in range(1,n)

        # this is dcolumn/dlayerVMR.  This is for the GEO-FTS project which
        # does analysis on layers
        derivativeLayer = np.matmul(derivative, level_to_layer)

    # put values into return class
    # it seems like the derivative should be zeroed out for levels not used.
    # Why?  Changing those VMR values will have zero impact on the calculated quantity.
    # but currently it is non-zero for all levels. 
    result = {
        'columnLayer': columnLayer,       
        'column': columnTotal,       
        'derivative': derivative,        
        'level_to_layer': level_to_layer, 
        'derivativeLayer': derivativeLayer,
        'columnAirLayer': columnAirLayer, 
        'columnAir': columnAirTotal
    } 

    return result


def calculate_xvmr(vmrIn, pressureIn, pressureMax=-1, pressureMin=-1, pwfLayer=None, pwfLevel=None,):
    # IDL_LEGACY_NOTE: This function calculate_xco2 is the same as calculate_xco2 function in TOOLS/calculate_xco2.pro file.

    # AT_LINE 13 src_ms-101820/TOOLS/add_column.pro add_column
    function_name = "calculate_xvmr: "

    # It is possible that pressure is two dimensionals with shape (65,1) .  We will shrink it to one (65,) .
    if len(pressureIn.shape) == 2 and pressureIn.shape[1] == 1:
        print(function_name, "pressureIn.shape = ", pressureIn.shape)
        pressureIn = np.reshape(pressureIn, (pressureIn.shape[0]))

    pmax = np.amax(pressureIn)
    if pressureMax != -1:
        pmax = pressureMax
    if pmax > np.amax(pressureIn):
        pmax = np.amax(pressureIn)

    pmin = np.amin(pressureIn)
    if pressureMin != -1:
        pmin = pressureMin
    if pmin < np.amin(pressureIn):
        pmin = np.amin(pressureIn)

    # ind = where(pmax+0.05 GE pressureIn AND pmin-0.05 LE pressureIn AND pressureIn GT -0.001)
    ind1 = np.where(pressureIn <= pmax+0.05)
    ind2 = np.where(pressureIn >= pmin-0.05)
    ind3 = np.where(pressureIn > -0.001)
    ind4 = np.intersect1d(ind1, ind2)
    ind = np.array(np.intersect1d(ind3, ind4))

    if np.amin(np.abs(pmin - pressureIn)) > 0.05:
        if np.amax(ind) < len(pressureIn):
            ind = np.append(ind[:], np.amax(ind)+1)

    if np.amin(np.abs(pmax - pressureIn)) > 0.05:
        if np.amin(ind) > 0:
            prepend = np.array([np.amin(ind)])
            ind = np.concatenate((prepend, np.array(ind)))

    if len(ind) != len(pressureIn):
        indvmr = ind
        if len(vmrIn) < len(pressureIn):
            indvmr = indvmr[:-1]

        xvmr, pwfLevel0, pwfLayer0 = calculate_xvmr(vmrIn=vmrIn[indvmr], pressureIn=pressureIn[ind], pressureMax=pressureMax, pressureMin=pressureMin, pwfLayer=pwfLayer, pwfLevel=pwfLevel)

        pwfLevel = pressureIn*0
        pwfLevel[ind] = pwfLevel0

        pwfLayer = pressureIn[1:]*0
        pwfLayer[ind[:-1]] = pwfLayer0
    else:
        layerIn = 0
        if len(vmrIn) < len(pressureIn):
            layerIn = 1

        if len(pressureIn) > 2:
            pressure = deepcopy(pressureIn)
            pressure[0] = pmax
            pressure[-1] = pmin
        else:
            pressure = np.array([pmax, pmin])

        if layerIn == 1:
            pressureLayer = (pressure[:-1] + pressure[1:])/2
            pressureLayerIn = (pressureIn[:-1] + pressureIn[1:])/2
            if len(vmrIn) > 1:
                vmr = spi.griddata((pressureLayerIn,), vmrIn, pressureLayer, method='linear')
                nearest_vmr = spi.griddata((pressureLayerIn,), vmrIn, pressureLayer, method='nearest')
                # replace nans from linear interpolation with nearest inpterolation results
                nans = np.where(np.isnan(vmr))
                vmr[nans] = nearest_vmr[nans]
            else:
                vmr = np.array(vmrIn, vmrIn)
        else:
            vmr = spi.griddata((pressureIn,), vmrIn, pressure, method='linear')
            nearest_vmr = spi.griddata((pressureIn,), vmrIn, pressure, method='nearest')
            # replace nans from linear interpolation with nearest inpterolation results
            nans = np.where(np.isnan(vmr))
            vmr[nans] = nearest_vmr[nans]

        nn = len(pressure)
        pwfLayer = pressure[:nn-1] - pressure[1:]
        pwfLayer = pwfLayer / sum(pwfLayer)

        pwfLevel = np.zeros(nn)
        pwfLevel[:nn-1] = pwfLayer
        pwfLevel[1:nn] = pwfLevel[1:nn] + pwfLayer    
        pwfLevel = pwfLevel / sum(pwfLevel)

        if len(vmrIn) == nn:
            xvmr = sum(pwfLevel * vmr)
        else:
            xvmr = sum(pwfLayer * vmr)

    return xvmr, pwfLevel, pwfLayer

# This replaces the obsolete utilMath.PythonInterpolateLinear
def _interpol_1d(i_vector, i_abscissaValues, i_abscissaResult, i_kind=None):
    if i_kind is None:
        i_kind = 'linear'

    interpfunc = spi.interp1d(i_abscissaValues, i_vector, kind=i_kind, fill_value="extrapolate")
    o_interpolatedArray = interpfunc(i_abscissaResult)

    return o_interpolatedArray

def _make_maps(pressureIn, i_levels, i_linearFlag=False, i_averageFlag=None):
    # Create maps to map from FM to retrieval grid and back.
    # Returns a structure, where
    #    toPars: map to retrieval grid (e.g. ret = maps.toPars @ fm)
    #    toState: map to FM grid (e.g. fm = maps.toState @ ret)
    function_name = "make_maps: "

    o_maps = None
    toPars = None
    toState = None

    if i_averageFlag is not None and i_averageFlag is True:
        min_pressure = np.min(pressureIn[i_levels])
        max_pressure = np.max(pressureIn[i_levels])

        ind = []
        for ii in range(len(pressureIn)):
            if (pressureIn[ii] > min_pressure and pressureIn[ii] <= max_pressure):
                ind.append(ii)

        n = len(pressureIn)
        toState = np.zeros(shape=(1, n), dtype=np.int32)
        toState[0, ind] = 1

        print(function_name, "i_averageFlag TRUE not implemented yet.")
        assert False
    else:
        pressure = pressureIn
        if not i_linearFlag:
            pressure = np.log(pressure)

        if pressure[1] < pressure[0]:
            pressure = -pressure

        # PYTHON_NOTE: It is possible that some values in i_levels may index passed the size of pressure.
        # The size of pressure may be 63 and one indices may be 64.
        any_values_greater_than_size = (i_levels > pressure.size).any()
        if any_values_greater_than_size:
            o_cleaned_retrievalParameters = _remove_indices_too_big(i_levels, pressure)
            # Reassign i_levels to o_cleaned_retrievalParameters so it will contain indices that are within size of pressure.
            i_levels = o_cleaned_retrievalParameters

        m = len(i_levels)
        n = len(pressure)

        toPars = np.zeros(shape=(n, m), dtype=np.float64)
        toState = np.zeros(shape=(m, n), dtype=np.float64)

        if i_levels[m-1] > len(pressure):
            i_levels[m-1] = len(pressure)

        retFreq = pressure[i_levels-1]
        freq = pressure

        # AT_LINE 88 TWPR_TOOLS/make_maps.pro
        m = len(retFreq)
        n = len(freq)

        num_elements_processed = 0

        # AT_LINE 90 TWPR_TOOLS/make_maps.pro
        for jj in range(0, m-1): # For the loop goes from 0 to m-1 since Python does not include end range.
            ind1 = np.where(freq >= retFreq[jj])[0]
            ind2 = np.where(freq <= retFreq[jj+1])[0]

            # AT_LINE 94 TWPR_TOOLS/make_maps.pro
            # PYTHON_NOTE: The for loop goes from min(ind1) to max(ind2)+1 since in Python, it does not include the end range.
            for kk in range(min(ind1), max(ind2) + 1):
                freq1 = retFreq[jj]
                freq2 = retFreq[jj+1]

                # Do a sanity check so we won't be dividing by zero with (freq2 - freq1) below.
                if freq1 == freq2:
                    print(function_name, 'Check retrieval lvls for duplicates ', i_levels)
                    assert False 

                coef = np.float32(freq2 - freq[kk]) / np.float32(freq2 - freq1)
                toState[jj, kk] = coef
                toState[jj+1, kk] = 1.0 - coef

                num_elements_processed = num_elements_processed + 1
            # end for kk in range(min(ind1),max(ind2)+1)
        # end for jj in range(0, m-1):

        # TODO: We should really do this here
        # toState = toState.T
        # toPars = toPars.T

        # IDL: 
        # a = toState
        # toPars = invert(transpose(a)##a)##transpose(a)

        # NOTE: due to toState having switched rows and columns.
        # If using toState.T and toPars.T the calculation will be identical to IDL:
        # toPars[:, :] = np.linalg.inv(a.T @ a) @ a.T

        # Keep using wrong toState and toPars for now        
        a = np.copy(toState)
        toPars[:, :] = a.T @ np.linalg.inv(a @ a.T)
    # end else part of if i_averageFlag is not None and i_averageFlag is True:

    o_maps = {
        'toPars': toPars,
        'toState': toState
    }

    return o_maps


def _remove_indices_too_big(indices_list, i_pressure):
    # To keep from crashing, we have to inspect all values in indices_list-1 so we don't index passed the size of i_pressure.
    # To do that, we make a copy of indices_list to dirty_Parameters and then checking dirty_Parameters to build a cleaned_retrievalParameters.

    # If there is at least one indices too big, we have to remove it.
    any_values_greater_than_size = (indices_list > i_pressure.size).any() 
    num_indices_removed = 0
    if any_values_greater_than_size:
        dirty_Parameters = deepcopy(indices_list)
        o_cleaned_retrievalParameters = []  # Start out with a fresh list.  All elements added here will be indices that will not index pass size of i_pressure. 
        for temp_ii in range(0, len(dirty_Parameters)):  # Do not use the ii as index as it is already used for the species_name loop.
            if (dirty_Parameters[temp_ii]-1 < i_pressure.size):
                #print(function_name, "INSPECTING_VARIABLES:ADDING_ELEMENT:temp_ii, dirty_Parameters[temp_ii]", temp_ii, dirty_Parameters[temp_ii])
                o_cleaned_retrievalParameters.append(dirty_Parameters[temp_ii])
            else:
                num_indices_removed += 1
                pass
                #print(function_name, "INSPECTING_VARIABLES:SKIPPING_ELEMENT:temp_ii, dirty_Parameters[temp_ii]", temp_ii, dirty_Parameters[temp_ii])
    else:
        o_cleaned_retrievalParameters = indices_list # Return the original list.

    return(np.asarray(o_cleaned_retrievalParameters))

import numpy as np
import re

# Function to reverse normalise the values of features in the value list section of LIME explanations
# args:
#    exp : the Explanation object for which normalisation must be reversed
#    sequenceSteps : the number of time steps in each time series sequence
#    scaler : the scaler that was used for normalising the data
# returns:
#    exp : the Explanation object with reverse normalised values in the feature values section
def reverse_normalise_exp_values(exp, data_columns, sequenceSteps, scaler):
    timepoints = np.arange(sequenceSteps - 1)
    sequencelist = []

    # Build time-series sequences from data in Explanation object
    for timepoint in timepoints:
        colcount = timepoint
        values_list = []
        for col in data_columns:
            values_list.append(exp.domain_mapper.feature_values[colcount])
            colcount = colcount + len(timepoints)
        sequencelist.append(values_list)

    # Invert the normalisation transformation and round to a single decimal
    exp_values = scaler.inverse_transform(sequencelist).round(1)
    # Add 0 to all values to sign-flip all -0 values to 0
    for timepoint in timepoints:
        exp_values = [x+0 for x in exp_values]

    # Zip the data for each timepoint back together in the format of the Explanation object
    exp_values = list(zip(*exp_values))
    exp_values = [str(item) for sublist in exp_values for item in sublist]
    exp.domain_mapper.feature_values = exp_values
    return exp


# Function to reverse normalise the values of features in the discretised weight list section of LIME explanations
# args:
#    exp : the Explanation object for which normalisation must be reversed
#    sequenceSteps : the number of time steps in each time series sequence
#    scaler : the scaler that was used for normalising the data
# returns:
#    exp : the Explanation object with reverse normalised discretised feature values in the feature weight section
def reverse_normalise_discr_values(exp, data_columns, sequenceSteps, scaler):
    timepoints = np.arange(sequenceSteps - 1)
    inverted_values = []

    # Create a list of all discretisation pairs per timepoint.
    for timepoint in timepoints:
        colcount = timepoint
        values_list = []
        for col in data_columns:
            # Parse the discretized feature name for any numerical values
            values = [float(s) for s in re.findall(r'[+-]?[\d]*[.][\d]+', exp.domain_mapper.discretized_feature_names[colcount])]
            # If a discretisation only has a single boundary, fill the other pair value with the same boundary
            if len(values) < 2:
                values = [values[0], values[0]]
            values_list.append(values)
            colcount = colcount + len(timepoints)

        # Split the pairs so inverse transformation can be performed
        values_list = np.hsplit(np.array(values_list), 2)
        boundaries_list = []

        for side in values_list:
            side = side.flatten()
            boundaries_list.append(side)

        # Invert the normalisation transformation and round to a single decimal
        exp_values = scaler.inverse_transform(boundaries_list).round(1)
        # Add 0 to all values to sign-flip all -0 values to 0
        for side in boundaries_list:
            exp_values = [x+0 for x in exp_values]

        # Stack the two sides back together into a pair
        stacked_values = np.stack((exp_values[0], exp_values[1]), axis=1)
        inverted_values.append(stacked_values)

    # Zip the lists per timepoint into the format used in the Explanation object
    zippedvalues = list(zip(*inverted_values))
    zippedvalues = [item for sublist in zippedvalues for item in sublist]

    discretized_feature_names_list = []

    for timepoint in timepoints:
        colcount = timepoint
        namelist = []

        for col in data_columns:
            name1 = re.findall(r'(\s\D+\s.*\s\D+\s)', exp.domain_mapper.discretized_feature_names[colcount])
            name2 = re.findall(r'([a-zA-Z].*\s\D+\s)', exp.domain_mapper.discretized_feature_names[colcount])
            name = ''
            if len(name1) > 0: # this means the format for this row is like 'x < SEX_t-0 <= y'
                name = str(zippedvalues[colcount][0]) + name1[0] + str(zippedvalues[colcount][1])
            else: # this case means format is like 'ESS1_t-0 >= x', thus name2 should be used
                name = name2[0] + str(zippedvalues[colcount][0])

            namelist.append(name)
            colcount = colcount + len(timepoints)
        discretized_feature_names_list.append(namelist)

    discretized_feature_names_list = list(zip(*discretized_feature_names_list))
    discretized_feature_names_list = [item for sublist in discretized_feature_names_list for item in sublist]
    exp.domain_mapper.discretized_feature_names = discretized_feature_names_list
    return exp


# Function to reverse normalise all feature values in a LIME explanation
# args:
#    exp : the Explanation object for which normalisation must be reversed
#    sequenceSteps : the number of time steps in each time series sequence
#    scaler : the scaler that was used for normalising the data
# returns:
#    exp : the Explanation object with reverse normalised feature values
def reverse_normalise_values(exp, data_columns, sequenceSteps, scaler):
    exp = reverse_normalise_exp_values(exp, data_columns, sequenceSteps, scaler)
    exp = reverse_normalise_discr_values(exp, data_columns, sequenceSteps, scaler)
    return exp
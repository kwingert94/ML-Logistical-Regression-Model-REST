def find_variables(axesNames):
    """
    This function takes in the axes of the model parameters and finds the column anmes.
    It returns a list of strings with the names
    """
    variables = []
    for i in range(0, len(axesNames[0])):
        variables.append(axesNames[0][i])
    variables.sort()
    return variables
def numeric_age(row):
    """numeric_age convert age to numeric for pandas row.apply

    Parameters
    ----------
    row : pd.row_

    Returns
    -------
    value for row entry
    """
    if isinstance(row.age, float):
        return row.age
    row.age = int("".join(filter(str.isdigit, row.age)))
    return float(row.age)

""" For an apply function in pandas
age categories needs to be a dictionary like this:
{"preweaning": [0, 21], "weaning": [21, 42], "postweaning": [42, 1000]}

"""

def categorize_ages(row, age_categories):
    if age_categories is None:
        row.age_category = "ND"
        return row.age_category
    row.age = numeric_age(row)  # convert
    for k in age_categories.keys():
        if (
            row.age >= age_categories[k][0]
            and row.age <= age_categories[k][1]
        ):
            row.age_category = k
    return row.age_category


    
from typing import Union

""" Helper functions for handling the age data
and categorizing it into groups (bins)

"""

def numeric_age(row):
    """numeric_age convert age to numeric for pandas row.apply
    or directly on a row of data.
    Parameters
    ----------
    row : pandas row with 'age' column

    Returns
    -------
    updated row with floating point (but int) age
    """
    if isinstance(row.age, float):
        return row
    row.age = int("".join(filter(str.isdigit, row.age)))
    row.age = float(row.age)
    return row


def numeric_age_from_data(age):
    """numeric_age convert age to numeric value

    Parameters
    ----------
    age : str

    Returns
    -------
    float
    """
    if isinstance(age, float):
        return age
    age = int("".join(filter(str.isdigit, age)))
    return float(age)


def categorize_ages(row, age_categories:Union[dict, None]):
    """categorize_ages bin an age (in row.age, as a float)
    into an age category.
    The age categories are (usually) pulled from the experiment configuraiton
    file, in the "age_categories" dictionary

    The dictionary looks like:
    {"preweaning": [0, 21], "weaning": [21, 42], "postweaning": [42, 1000]}
    
    Parameters
    ----------
    row : pandas series with 'age' column
        row for one cell of data
    age_categories : Union[dict, None]
        the age categories devined as above.

    Returns
    -------
    the ROW
        _description_
    """
    if age_categories is None:
        row.age_category = 0.
        return row.age_category
    row = numeric_age(row)  # convert to a floating number in row.age
    for k in age_categories.keys():
        if row.age >= age_categories[k][0] and row.age <= age_categories[k][1]:
            row.age_category = k
    return row


def get_age_category(age, age_categories):
    # given an age string, return the category that fits
    # from the age_categories dictionary
    n_age = numeric_age_from_data(age)
    for k in age_categories.keys():
        if n_age >= age_categories[k][0] and n_age <= age_categories[k][1]:
            return k
    return "ND"

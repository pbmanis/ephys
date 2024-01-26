""" get_computer
Get the name of the current computer that we are running on
Returns
-------
str:
    computer name
"""
import os
import subprocess


def get_computer():
    if os.name == "nt":
        computer_name = os.environ["COMPUTERNAME"]
    else:
        computer_name = subprocess.getoutput("scutil --get ComputerName")
    return computer_name


from __future__ import division

import sys
import os

organization_folder = os.environ['ORGANIZATION']

# this_script_dir = os.path.dirname(os.path.realpath(__file__))
# pyperclip_dir = os.path.join(organization_folder, "Programs\\Python\\pyperclip-1.5.11\\pyperclip")
# sys.path.append(pyperclip_dir)

import pyperclip

def get_from_clipboard()  ->  str:
    return pyperclip.paste()

def put_on_clipboard(string)  ->  None:
    return pyperclip.copy(string)

def modify_clipboard(fcn)  ->  None:
    put_on_clipboard(fcn(get_from_clipboard()))
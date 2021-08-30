#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:51:12 2018

@author: walther
"""
import sys
import textwrap
import warnings



### setting global intention level and tab
glvl=0
gtab='  |'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if sys.version_info[0]==3:
    def _warning(
        message,
        category = UserWarning,
        filename = '',
        lineno = -1,
        file='',
        line=''):
        """Replace the default warnings.showwarning function to include in print_status
        """
        msg = warnings.WarningMessage(message, category, filename, lineno, file, line)
        print_status('There is a warning!:',glvl,gtab,'>>','warning')
        print_status(str(msg.message),glvl,gtab,'>>','warning')
    warnings.showwarning = _warning
else:
    def _warning(
        message,
        category = UserWarning,
        filename = '',
        lineno = -1):
        """Replace the default warnings.showwarning function to include in print_status
        """
        print_status('There is a warning!:',glvl,gtab,'>>','warning')
        print_status(message[0],glvl,gtab,'>>','warning')
    warnings.showwarning = _warning


def print_status(txt,lvl=0,tab='  |',pfx='',style='',end='',flush=False):
    r"""
    Print `txt` to stout with defined intention level for easy looks ;)
    
    Parameters
    ----------
    txt : str
        `txt` represents the text to print to stout.
    lvl : int, optional
        `lvl` represents the intention level of the message to print. It sets
        the global `glvl` variable.
        default: 0
    tab : str, optional
        `tab` represents the prefix string which is repeated `lvl` times. It
        sets the global `gtab` variable.
        default: '  |'
    pfx : str, optional
        `pfx` represents the prefix printed directly before `txt`.
        default: ''
    style : str, optional
        `style` choose the style of `txt`. Default is None.
        It can be anything of  ['blue' or 'b',
                                'green' or 'g',
                                'fail',
                                'warning',
                                'bold',
                                'header',
                                'underline']
    """
    if lvl<0:
        return 0
    global glvl
    glvl=lvl
    offset=''
    lines=textwrap.wrap(txt,width=69-lvl*len(tab)-len(pfx),break_long_words=False)
    for i,l in enumerate(lines):
        if i!=0:
            offset='  '
        if style.lower()=='blue' or style.lower()=='b':
            pl=bcolors.OKBLUE+l+bcolors.ENDC
        elif style.lower()=='green' or style.lower()=='g':
            pl=bcolors.OKGREEN+l+bcolors.ENDC
        elif style.lower()=='fail': 
            pl=bcolors.FAIL+l+bcolors.ENDC
        elif style.lower()=='warning':
            pl=bcolors.WARNING+l+bcolors.ENDC 
        elif style.lower()=='bold':
            pl=bcolors.BOLD+l+bcolors.ENDC     
        elif style.lower()=='header':
            pl=bcolors.HEADER+l+bcolors.ENDC  
        elif style.lower()=='underline':
            pl=bcolors.UNDERLINE+l+bcolors.ENDC
        else:
            pl=l
        print(lvl*tab+pfx+offset+pl+end)
        if flush:
            sys.stdout.flush()
    return 0



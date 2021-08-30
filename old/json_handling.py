#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 08:50:59 2018

@author: walther
"""
import json

def merge_default(cnf):
    """
    This function merges the default dictionary with all key dictionaries, replacing default values if needed
    input:
        cnf - dict - one key has to be 'default' (cnf['default'] has to be a dict)
    output:
        cnf - dict - cnf['default'] is deleted, cnf[cnf.keys()] are merged with cnf['default']
    example:
        IN :merge_default({'default':{'a':1,'b':2},'var':{'b':3,'c':4}})
        OUT:{'var': {'a': 1, 'b': 3, 'c': 4}}
    """
    default = cnf.pop('default') # get the default and remove it from the cnf dictionary
    for key in cnf.keys():
        temp=default.copy()
        temp.update(cnf[key]) #merge and overwrite default
        cnf[key]=temp #replace with the merged dictionary
    return cnf

def get_item(ipath,idata):
    """
    This function look extract a certain value from the dict idata from the key path ipath.
    input:
        ipath - string - the key path to search in the dict idata - example: 'instruments/Pyr1/unit' -> idata['instruments']['Pyr1']['unit']
        idata - dict - dictionary with lots of data
    output:
        {key:value} - key refers to the last key in ipath; value is the value in idata(ipath)
    example:
        In :get_item('instruments/Pyr1/unit',{instruments:{'Pyr1':{'unit':'Wm-2'},'Resistor':{'unit':'Ohm'}}})
        Out:{'unit':'Wm-2'}
    """
    paths = ipath.split("/")
    data = idata
    for i in range(0,len(paths)):
        data = data[paths[i]]
    return {paths[-1]:data}

def follow_ref(obj,fn,link_key):
    """
    The object_hook function for the json module. It passes the original object unless the key==link_key, then it look up the link to internal key or key of an external json file.
    input:
        obj - dict - obj[link_key] should be in the form: 'filepath#dictpath' (e.g.: "cnf.json#instruments/Pyr1/unit" -> see get_item)
                -- filepath  - string - can be absolute, relative (link to external json file) or empty (link to internal key)
        fn  - string - path to the original json file
        link_key - string - string to trigger the lookup of the link stored in obj[link_key]
    output:
        obj - dict - the original obj, but obj[link_key] is replaced by the values found in the link obj[link_key]
    """
    if '$ref' in obj.keys():
        rfile,ipath=obj[link_key].split('#')
        if len(rfile)==0: # intenal link (to json file in fn)
            tcnf=load_json(fn)
        else: # external link to a json file
            if rfile[0]=='/':#absolute path
                tcnf=load_json(rfile,follow_ref)
            else:
                tcnf=load_json(fn[:fn.rfind('/')+1]+rfile,follow_ref) # relative to the original json file in fn
        if 'default' in tcnf.keys():# if nessesary merge default values before extracting the desired item
            tcnf=merge_default(tcnf)
        obj=get_item(ipath,tcnf)
    return obj

def load_json(fn,link_key='$ref'):
    """
    utilize the json module to load the config json file
     - the object_hook funktion is used to enable linking of of internal and external values (see follow_ref). 
     - if the json file contains a default key -> the default values are merged into all values (see merge_default)
    input:
        fn - string - path to the json file
        link_key - string - the key for the json dictionary to trigger following the link to another value (default:'$ref') (see follow_ref)
    output:
        cnf - dict - the dictionary parsed from the json file
    """
    with open(fn,'r') as f:
        cnf=json.load(f,object_hook=lambda x:follow_ref(x,fn,link_key))
    if 'default' in cnf.keys():
        cnf=merge_default(cnf)
    return cnf


    


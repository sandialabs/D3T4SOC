#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import yaml
import os
import shutil

class Parser:
    """
        Parser to read yaml file
    """
    
    def __init__(self):
        # self.inyaml = inyaml
        
        # Argument parser setup
        parser = argparse.ArgumentParser(description="Read YAML config file as input.")
        parser.add_argument(
                            '--input',
                            type=str,
                            default='parameters.yaml',
                            help='Path to the YAML parameter file (default: parameters.yaml)'
                            )
        self.args = parser.parse_args()
        self.params = self.__read_yaml__()
           
    def __read_yaml__(self):
        # Read YAML file
        with open(self.args.input, 'r') as f:
            params = yaml.safe_load(f)
        return params
    
    def save_parameters(self, outfolder):
        outfile = os.path.join(outfolder,'parameters.yaml')
        # with open(outfile, 'w') as f:
        #     yaml.safe_dump(self.params, f)
        shutil.copy(self.args.input, outfile)
            
        
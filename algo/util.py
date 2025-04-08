#-------------------------------------------------------------------------------
# @description
#  common classes
#
# @author 
#  Mengsu Ding<dingmengsu.dms@alibaba-inc.com>
#
#-------------------------------------------------------------------------------

from typing import List
import unittest
import json


class Config:
    def __init__(self, conf_file: str = None) -> None:
        self.model_inprice_dict:dict[str, float]  = {}
        self.model_outprice_dict:dict[str, float] = {}
        self.budget    = None
        self.num_steps = None

        num_steps: int = None
        model_list: List[str]      = None
        inprice_list: List[float]  = None
        outprice_list: List[float] = None

   
    def set_num_steps(self, num_steps: int):
        self.num_steps = num_steps
    
    def set_model_price(self, model: str, inprice: int, outprice: int) -> None:
        self.model_inprice_dict[model] = inprice
        self.model_outprice_dict[model] = outprice

    def get_model_price(self, model: str):
        """ Return the unit price for input/1K and output/1K. """
        inprice: float = self.model_inprice_dict[model]
        outprice: float = self.model_outprice_dict[model]
        return inprice, outprice


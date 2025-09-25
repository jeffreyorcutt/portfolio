import re

 # setup the Regex Patterns in an object and compile them once to save on processing       
class RegexPatterns:
    def __init__(self):
        self.pattern_1899_date = re.compile(r'([1][2]\/[3][0]\/[1][8][9][9]) (\d{1,2}:\d{2})(.*)')
        self.pattern_yr_first_date = re.compile(r'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})(.*)')
        self.pattern_thai_date = re.compile(r'(\d{1,2}\/\d{1,2}\/[2][5][6].) (\d{1,2}:\d{2})(.*)')
        self.pattern_every_date = re.compile(r'(\d{1,2}\/\d{1,2}\/\d{4}) (\d{1,2}:\d{2})(.*)')
        self.pattern_zero_date = re.compile(r'(\d{1,2}\/\d{1,2}\/\d{4}) (\d{1,2}:\d{2})(.*)')
        
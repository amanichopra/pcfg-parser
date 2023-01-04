"""
COMS W4705 - Natural Language Processing - Summer 2022 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose
from warnings import warn

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        for lhs_symbol in self.lhs_to_rules:
            if not lhs_symbol.isupper():
                warn(f'Invalid grammar due to "{lhs_symbol}" non-terminal symbol being lowercase!')
                return False

            rhs_probs = [rhs[2] for rhs in self.lhs_to_rules[lhs_symbol]]
            if not isclose(sum(rhs_probs), 1):
                warn(f'Invalid grammar due to probabilities for all rules with same "{lhs_symbol}" symbol not summing to 1!')
                return False

            rhs_symbols = [rhs[1] for rhs in self.lhs_to_rules[lhs_symbol]]
            for symbols in rhs_symbols:
                if len(symbols) == 1 and symbols[0].isupper():
                    warn(f'Rule: "{lhs_symbol} -> {symbols[0]}" written as terminal but "{symbols[0]}" is not lowercase!')
                    return False
                elif len(symbols) == 2 and not (symbols[0].isupper() and symbols[1].isupper()):
                    warn(f'Rule: "{lhs_symbol} -> {symbols[0]} {symbols[1]}" written as non-terminal but "{symbols[0]}" and "{symbols[1]}" are not uppercase!')
                    return False
                elif len(symbols) > 2:
                    warn(f'Rule: "{lhs_symbol} -> {" ".join(symbols)}" has more than 2 symbols in its LHS!')
                    return False
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        default_grammar_file = open('./atis3.pcfg')
        grammar = Pcfg(default_grammar_file)
        default_grammar_file.close()

    else:
        grammar_file = open(sys.argv[1])
        grammar = Pcfg(grammar_file)
        grammar_file.close()

    if grammar.verify_grammar():
        print('Grammar verified to be a valid PCFG in CNF!')
    else:
        print('Grammar is invalid PCFG!')
        

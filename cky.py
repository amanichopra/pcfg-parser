"""
COMS W4705 - Natural Language Processing - Summer 2022
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg
from copy import deepcopy


### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[2], str) and isinstance(bp[0], int) and isinstance(bp[1], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True


def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True


def test_parser():
    print(f'Running tests on parser...')
    with open('cfg_test.pcfg') as grammar_file:
        grammar = Pcfg(grammar_file)
        if not grammar.verify_grammar():
            raise Exception('Grammar is invalid PCFG! Check "cfg_test.pcfg"!')

        parser = CkyParser(grammar)

        tokens = ['the', 'rain', 'rains', 'down']
        dp = parser.cky_membership_checking_dp(tokens)
        assert dp[0][0] == set(['D'])
        assert dp[0][1] == set(['NP'])
        assert dp[0][2] == set(['S'])
        assert dp[0][3] == set(['S'])
        assert dp[1][1] == set(['NP', 'N', 'V'])
        assert dp[1][2] == set(['S'])
        assert dp[1][3] == set(['S'])
        assert dp[2][2] == set(['N', 'V'])
        assert dp[2][3] == set(['VP'])
        assert dp[3][3] == set(['ADV'])

        for i in range(len(tokens)):
            for j in range(i):
                assert len(dp[i][j]) == 0

        assert parser.is_in_language(tokens)

        backpointers, probs = parser.parse_with_backpointers(tokens)
        assert check_table_format(backpointers)
        assert check_probs_format(probs)

        tokens = ['rain', 'rain']
        dp = parser.cky_membership_checking_dp(tokens)
        assert dp[0][0] == set(['NP', 'V', 'N'])
        assert dp[0][1] == set(['S', 'VP'])
        assert dp[1][1] == set(['NP', 'V', 'N'])

        assert len(dp[1][0]) == 0

        assert parser.is_in_language(tokens)

        tokens = ['hello']
        assert not parser.is_in_language(tokens)

        tokens = ['down']
        assert not parser.is_in_language(tokens)


    print('Tests sucessfully complete!')
    print()


class BackPointerRule():
    def __init__(self, lhs, lhs_span, rhs1, rhs1_span, rhs2=None, rhs2_span=None):
        self.lhs = lhs
        self.lhs_span = lhs_span
        self.rhs1 = rhs1
        self.rhs1_span = rhs1_span
        self.rhs2 = rhs2
        self.rhs2_span = rhs2_span

    def __repr__(self):
        if self.rhs2:
            return f'{self.lhs}{self.lhs_span} -> {self.rhs1}{self.rhs1_span}, {self.rhs2}{self.rhs2_span}'
        else:
            return f'{self.lhs}{self.lhs_span} -> {self.rhs1}{self.rhs1_span}'


class ParseTreeNode():
    def __init__(self, symbol=None):
        self.symbol = symbol
        self.left = None
        self.right = None

    def is_leaf(self):
        if not (self.left and self.right) and type(self.symbol) == ParseTreeLeaf:
            return True
        return False

    def get_tree_repr(self):
        tree = self._build_tree_repr(self)
        return tree

    def _build_tree_repr(self, tree, tree_repr=[]):
        if tree.is_leaf():
            tree_repr.append(tree.symbol.symbol)
            tree_repr.append(tree.symbol.terminal)
            return
        else:
            tree_repr.append(tree.symbol)

            tree_repr.append([])
            self._build_tree_repr(tree.left, tree_repr[-1])

            tree_repr.append([])
            self._build_tree_repr(tree.right, tree_repr[-1])

        tree_repr[1] = tuple(tree_repr[1])
        tree_repr[2] = tuple(tree_repr[2])
        return tuple(tree_repr)


class ParseTreeLeaf():
    def __init__(self, symbol, terminal):
        self.symbol = symbol
        self.terminal = terminal


class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def cky_membership_checking_dp(self, tokens):
        # initalize dp table
        dp = [[set() for j in range(len(tokens))] for i in range(len(tokens))]

        # base cases
        for i in range(len(tokens)):
            token = tuple([tokens[i]])
            token_describers = set([describers[0] for describers in self.grammar.rhs_to_rules[token]])
            dp[i][i] = dp[i][i].union(token_describers)

        for length in range(2, len(tokens) + 1):
            for i in range(len(tokens)):
                j = i + length - 1
                if j == len(tokens):
                    break
                for k in range(length - 1):
                    symbs = self.get_all_lhs_for_all_rhs(dp[i][j - k - 1], dp[i + length - 1 - k][j])
                    dp[i][j] = dp[i][j].union(symbs)
        return dp

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        dp = self.cky_membership_checking_dp(tokens)
        if self.grammar.startsymbol in dp[0][len(tokens)-1]:
            return True
        else:
            return False

    def get_all_lhs_for_all_rhs(self, rhs1, rhs2):
        """
        Find all the lhs symbols that can produce any combination of the rhs symbols.
        """
        symbs = set()
        for s1 in rhs1:
            for s2 in rhs2:
                symb_pair = tuple([s1, s2])
                lhs_symbs = set([rule[0] for rule in self.grammar.rhs_to_rules[symb_pair]])
                symbs = symbs.union(lhs_symbs)
        return symbs

    def parse_with_backpointers(self, tokens, log_prob=True):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        # initalize dp table
        dp = {(i, i): {} for i in range(len(tokens))}

        # initialize backpointer table
        backpointers = {(i, i): {} for i in range(len(tokens))}

        # base cases for dp table
        for i in range(len(tokens)):
            rules = self.grammar.rhs_to_rules[tuple([tokens[i]])]
            for rule in rules:
                if log_prob:
                    dp[(i, i)][rule[0]] = math.log2(rule[2])
                else:
                    dp[(i, i)][rule[0]] = rule[2]

        # base cases for backpointer table
        for i in range(len(tokens)):
            rules = self.grammar.rhs_to_rules[tuple([tokens[i]])]
            for rule in rules:
                backpointers[(i, i)][rule[0]] = ((i, i, rule[1][0]), (i, i, rule[1][0]))

        for length in range(2, len(tokens) + 1):
            for i in range(len(tokens)):
                j = i + length - 1
                if j == len(tokens):
                    break
                maxes = {}
                best_backpointers = {}
                for k in range(length - 1):
                    for t1 in dp[(i, j - k - 1)].keys():
                        for t2 in dp[(i + length - 1 - k, j)].keys():
                            t_pair = tuple([t1, t2])
                            for rule in self.grammar.rhs_to_rules[t_pair]:
                                if rule[0] not in maxes:
                                    if log_prob:
                                        maxes[rule[0]] = math.log2(rule[2]) + dp[(i, j - k - 1)][t1] + dp[(i + length - 1 - k, j)][t2]
                                    else:
                                        maxes[rule[0]] = rule[2] * dp[(i, j - k - 1)][t1] * dp[(i + length - 1 - k, j)][t2]
                                    best_backpointers[rule[0]] = ((i, j - k - 1, t1), (i + length - 1 - k, j, t2))

                                elif math.log2(rule[2]) + dp[(i, j - k - 1)][t1] + dp[(i + length - 1 - k, j)][t2] > maxes[rule[0]]:
                                    if log_prob:
                                        maxes[rule[0]] = math.log2(rule[2]) + dp[(i, j - k - 1)][t1] + dp[(i + length - 1 - k, j)][t2]
                                    else:
                                        maxes[rule[0]] = rule[2] * dp[(i, j - k - 1)][t1] * dp[(i + length - 1 - k, j)][t2]
                                    best_backpointers[rule[0]] = ((i, j - k - 1, t1), (i + length - 1 - k, j, t2))
                dp[(i, j)] = maxes
                backpointers[(i, j)] = best_backpointers
        return backpointers, dp


def build_tree(chart, i, j, nt, tree=ParseTreeNode()):
    if nt not in chart[(i, j - 1)]:
        raise KeyError(f'No parse tree rooted in non-terminal {nt} covering span ({i}, {j})!')

    if i == j - 1:
        symbol = nt
        terminal = chart[i, j - 1][nt][0][-1]
        tree.symbol = ParseTreeLeaf(symbol, terminal)
        return
    else:
        tree.symbol = nt

        tree.left = ParseTreeNode()
        build_tree(chart, chart[i, j - 1][nt][0][0], chart[i, j - 1][nt][0][1] + 1, chart[i, j - 1][nt][0][2], tree=tree.left)

        tree.right = ParseTreeNode()
        build_tree(chart, chart[i, j - 1][nt][1][0], chart[i, j - 1][nt][1][1] + 1, chart[i, j - 1][nt][1][2], tree=tree.right)

    return tree


def get_tree_rep(tree, tree_repr=None):
    if tree_repr is None:
        tree_repr = []
    if tree.is_leaf():
        tree_repr.append(tree.symbol.symbol)
        tree_repr.append(tree.symbol.terminal)
        return
    else:
        tree_repr.append(tree.symbol)

        tree_repr.append([])
        get_tree_rep(tree.left, tree_repr[-1])

        tree_repr.append([])
        get_tree_rep(tree.right, tree_repr[-1])

    tree_repr[1] = tuple(tree_repr[1])
    tree_repr[2] = tuple(tree_repr[2])
    return tuple(tree_repr)


def get_tree(chart, i, j, nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    tree = build_tree(chart, i, j, nt)
    return get_tree_rep(tree)


if __name__ == "__main__":
    test_parser()

    grammar_test_file = 'atis3.pcfg'
    with open(grammar_test_file,'r') as grammar_file:
        grammar = Pcfg(grammar_file)

        # verify grammar
        if grammar.verify_grammar():
            print('Grammar verified to be a valid PCFG in CNF!')
        else:
            print('Grammar is invalid PCFG!')
        print()

        parser = CkyParser(grammar)
        toks = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(f'Checking if sentence "{" ".join(toks)}" is in language...')
        if parser.is_in_language(toks):
            print('Sentence is in language!')
        else:
            print('Sentence is not in language!')
        print()

        print(f'Running parser on sentence "{" ".join(toks)}"...')
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        print()

        parse_tree_nt_root = grammar.startsymbol
        span_start = 0
        span_end = len(toks)
        print(f'Retrieving parse tree rooted at {parse_tree_nt_root} covering the span ({span_start}, {span_end})...')
        parse_tree = get_tree(table, span_start, span_end, parse_tree_nt_root)
        print(f'Parse tree looks like below:')
        print(parse_tree)
        

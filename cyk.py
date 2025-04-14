#CYK
from collections import defaultdict
from nltk.tree import Tree
from tabulate import tabulate

# ---------------------------
# READ GRAMMAR FROM FILE
# Format: A -> B C OR A -> word
# ---------------------------
def read_grammar(filename):
    grammar = defaultdict(list)
    rhs_to_lhs = defaultdict(set)
    with open(filename) as f:
        for line in f:
            if '->' not in line:
                continue
            lhs, rhs = line.strip().split('->')
            lhs = lhs.strip()
            rhs_symbols = rhs.strip().split()
            grammar[lhs].append(rhs_symbols)
            rhs_to_lhs[tuple(rhs_symbols)].add(lhs)
    return grammar, rhs_to_lhs

# ---------------------------
# CYK PARSER WITH BACKPOINTERS
# ---------------------------
def cyk_parser(tokens, rhs_to_lhs):
    n = len(tokens)
    table = [[set() for _ in range(n)] for _ in range(n)]
    back = [[defaultdict(list) for _ in range(n)] for _ in range(n)]

    # Fill diagonals
    for i, token in enumerate(tokens):
        for lhs in rhs_to_lhs.get((token,), []):
            table[i][i].add(lhs)
            print(f"Matched terminal: {lhs}[1,{i+1}] -> {token}")

    # Fill upper triangle
    for l in range(2, n + 1):  # Span length
        for i in range(n - l + 1):  # Start index
            j = i + l - 1
            for k in range(i, j):
                for B in table[i][k]:
                    for C in table[k+1][j]:
                        for A in rhs_to_lhs.get((B, C), []):
                            table[i][j].add(A)
                            back[i][j][A].append((k, B, C))
                            print(f"Applied Rule: {A}[{l},{i+1}] --> {B}[{k-i+1},{i+1}] {C}[{j-k},{k+2}]")

    # Tree building
    def build_tree(i, j, symbol):
        if i == j:
            return (symbol, tokens[i])
        for k, B, C in back[i][j].get(symbol, []):
            left = build_tree(i, k, B)
            right = build_tree(k+1, j, C)
            return (symbol, left, right)

    return table, back, build_tree if 'S' in table[0][n-1] else None

# ---------------------------
# CONVERT TO nltk.Tree FOR DISPLAY
# ---------------------------
def tuple_to_nltk_tree(tree_tuple):
    if isinstance(tree_tuple, tuple):
        label = tree_tuple[0]
        children = [tuple_to_nltk_tree(child) for child in tree_tuple[1:]]
        return Tree(label, children)
    else:
        return tree_tuple

# ---------------------------
# PRINT CYK TABLE USING TABULATE
# ---------------------------
def print_table(table, tokens):
    n = len(tokens)
    display_table = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if table[i][j]:
                display_table[i][j] = ", ".join(sorted(table[i][j]))
    headers = [f"{i+1}:{w}" for i, w in enumerate(tokens)]
    print("\nCYK Parse Table:\n")
    print(tabulate(display_table, headers=headers, tablefmt="grid"))

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    grammar_file = "/content/cyk_grammar.txt"  # Replace with your file path
    sentence = "astronomers saw stars with ears"
    tokens = sentence.split()

    grammar, rhs_to_lhs = read_grammar(grammar_file)
    table, back, tree_builder = cyk_parser(tokens, rhs_to_lhs)

    if tree_builder:
        tree_tuple = tree_builder(0, len(tokens)-1, 'S')
        print("\n✔ Sentence is valid.\n")
        nltk_tree = tuple_to_nltk_tree(tree_tuple)
        nltk_tree.pretty_print()
    else:
        print("\n✘ Sentence is invalid according to the grammar.\n")

    print_table(table,tokens)

from collections import defaultdict
import math
import pprint
from prettytable import PrettyTable

# Step 1: Hardcoded parse trees from the image
tree1 = ['S',
           ['NP', 'John'],
           ['VP',
               ['VP', ['V', 'called'], ['NP', 'Mary']],
               ['PP', ['P', 'from'], ['NP', 'Denver']]
           ]
        ]

tree2 = ['S',
           ['NP', 'John'],
           ['VP',
               ['V', 'called'],
               ['NP',
                   ['NP', 'Mary'],
                   ['PP', ['P', 'from'], ['NP', 'Denver']]
               ]
           ]
        ]

# Step 2: Extract productions
def extract_productions(tree, productions):
    if isinstance(tree, list):
        lhs = tree[0]
        rhs = []

        for child in tree[1:]:
            if isinstance(child, list):
                rhs.append(child[0])
                extract_productions(child, productions)
            else:
                rhs.append(child)
        rule = (lhs, tuple(rhs))
        productions[rule] += 1

productions = defaultdict(int)
extract_productions(tree1, productions)
extract_productions(tree2, productions)

# Step 3: Compute PCFG
lhs_counts = defaultdict(int)
for (lhs, rhs), count in productions.items():
    lhs_counts[lhs] += count

pcfg = defaultdict(list)
for (lhs, rhs), count in productions.items():
    prob = count / lhs_counts[lhs]
    pcfg[lhs].append((rhs, prob))

print("\n--- PCFG ---")
for lhs, rules in pcfg.items():
    for rhs, prob in rules:
        print(f"{lhs} -> {' '.join(rhs)} [{prob:.2f}]")


# Step 4: Viterbi parser for new sentence
sentence = ['John', 'called', 'Mary', 'from', 'Denver']
n = len(sentence)
table = [[defaultdict(lambda: (-math.inf, None)) for _ in range(n)] for _ in range(n)]

# Step 5: Initialize terminals
for i, word in enumerate(sentence):
    for lhs, rules in pcfg.items():
        for rhs, prob in rules:
            if len(rhs) == 1 and rhs[0] == word:
                table[i][i][lhs] = (math.log(prob), word)

# Step 6: CKY-style Viterbi algorithm
for span in range(2, n+1):
    for i in range(n - span + 1):
        j = i + span - 1
        for k in range(i, j):
            for lhs, rules in pcfg.items():
                for rhs, prob in rules:
                    if len(rhs) == 2:
                        B, C = rhs
                        if B in table[i][k] and C in table[k+1][j]:
                            prob_B, back_B = table[i][k][B]
                            prob_C, back_C = table[k+1][j][C]
                            total_prob = math.log(prob) + prob_B + prob_C
                            if total_prob > table[i][j][lhs][0]:
                                table[i][j][lhs] = (total_prob, (k, B, C))

# Step 7: Backtrack to recover tree
def build_tree(i, j, symbol):
    prob, back = table[i][j].get(symbol, (-math.inf, None))

    # Terminal rule: back is a string (the word)
    if isinstance(back, str):
        return [symbol, back]

    # Unary case fallback (not expected here, but safe)
    if back is None:
        return [symbol]

    # Binary rule: back = (k, B, C)
    k, B, C = back
    left = build_tree(i, k, B)
    right = build_tree(k+1, j, C)
    return [symbol, left, right]

# Final output
print("\n--- Most Probable Parse Tree ---")
if 'S' in table[0][n-1]:
    tree = build_tree(0, n-1, 'S')
    pprint.pprint(tree)

    print(f"\nProbability of the parse tree: {math.exp(table[0][n-1]['S'][0]):.8f}")

    print("\n--- Viterbi Parsing Table (Triangular Format) ---\n")

    pretty_table = []

    for row in range(n):
        current_row = []
        for col in range(n):
            if col < row:
                current_row.append("")  # lower triangle blank
            else:
                cell = table[row][col]
                if not cell:
                    current_row.append("-")
                else:
                    entries = []
                    for symbol, (prob, back) in sorted(cell.items(), key=lambda x: -x[1][0]):
                        prob_val = math.exp(prob)
                        if isinstance(back, str):
                            entries.append(f"{symbol}({prob_val:.2f})")
                        else:
                            entries.append(f"{symbol}({prob_val:.6f})")
                    current_row.append("\n".join(entries))
        pretty_table.append(current_row)

    # Create header row
    headers = [""] + sentence
    cyk = PrettyTable()
    cyk.field_names = headers
    for i, row in enumerate(pretty_table):
        cyk.add_row([sentence[i]] + row)

    print(cyk)

else:
    print("No valid parse found.")

# IF YOU HAVE 5-6 TREES IN QUESTION PAPER, DO LIKE THIS
#LOOP
# trees = [
#     ['S',
#         ['NP', 'John'],
#         ['VP',
#             ['VP', ['V', 'called'], ['NP', 'Mary']],
#             ['PP', ['P', 'from'], ['NP', 'Denver']]
#         ]
#     ],
#     ['S',
#         ['NP', 'John'],
#         ['VP',
#             ['V', 'called'],
#             ['NP',
#                 ['NP', 'Mary'],
#                 ['PP', ['P', 'from'], ['NP', 'Denver']]
#             ]
#         ]
#     ],
#     # Add 4 more trees here:
#     tree3,
#     tree4,
#     tree5,
#     tree6,
# ]

# CHANGE IN THIS PART

# productions = defaultdict(int)

# for tree in trees:
#     extract_productions(tree, productions)


# #TO CHECK PRODUCTIONS
# print("\n--- Production Counts ---")
# for (lhs, rhs), count in productions.items():
#     print(f"{lhs} -> {' '.join(rhs)} : {count}")

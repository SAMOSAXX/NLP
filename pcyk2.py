
from collections import defaultdict
import math
import pprint

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

    # Calculate the maximum width needed for each cell
    max_cell_width = 0
    for i in range(n):
        for j in range(i, n):
            cell = table[i][j]
            if cell:
                cell_str = ""
                for symbol, (prob, back) in sorted(cell.items(), key=lambda x: -x[1][0]):
                    prob_val = math.exp(prob)
                    if isinstance(back, str):
                        cell_str += f"{symbol}({prob_val:.2f})\n"
                    else:
                        cell_str += f"{symbol}({prob_val:.6f})\n"
                max_cell_width = max(max_cell_width, max(len(line) for line in cell_str.split('\n')))

    # Add some padding
    cell_width = max(max_cell_width + 2, 15)
    
    # Print header
    header = ' ' * 10
    for word in sentence:
        header += word.center(cell_width)
    print(header)
    
    # Print separator
    print('-' * (10 + cell_width * n))
    
    # Print table rows
    for i in range(n):
        row = sentence[i].ljust(10)
        for j in range(n):
            if j < i:
                row += ' ' * cell_width  # Empty cell for lower triangle
            else:
                cell = table[i][j]
                if not cell:
                    row += '-'.center(cell_width)
                else:
                    cell_content = []
                    for symbol, (prob, back) in sorted(cell.items(), key=lambda x: -x[1][0]):
                        prob_val = math.exp(prob)
                        if isinstance(back, str):
                            cell_content.append(f"{symbol}({prob_val:.2f})")
                        else:
                            cell_content.append(f"{symbol}({prob_val:.6f})")
                    
                    # Join multiple entries with newlines and padding
                    cell_str = '\n'.join(cell_content)
                    
                    # Split the cell content by lines to pad each line
                    cell_lines = cell_str.split('\n')
                    padded_lines = [line.center(cell_width) for line in cell_lines]
                    
                    # For the first row, we'll just use the first line
                    row += padded_lines[0] if padded_lines else ' ' * cell_width
        
        print(row)
        
        # Print additional lines for cells with multiple entries
        max_lines = 0
        for j in range(i, n):
            cell = table[i][j]
            if cell:
                cell_content = []
                for symbol, (prob, back) in sorted(cell.items(), key=lambda x: -x[1][0]):
                    cell_content.append(symbol)
                max_lines = max(max_lines, len(cell_content))
        
        # Print additional lines if needed
        for line_idx in range(1, max_lines):  # Start from 1 as we already printed the first line
            multi_line_row = ' ' * 10
            for j in range(n):
                if j < i:
                    multi_line_row += ' ' * cell_width
                else:
                    cell = table[i][j]
                    if not cell:
                        multi_line_row += ' ' * cell_width
                    else:
                        cell_content = []
                        for symbol, (prob, back) in sorted(cell.items(), key=lambda x: -x[1][0]):
                            prob_val = math.exp(prob)
                            if isinstance(back, str):
                                cell_content.append(f"{symbol}({prob_val:.2f})")
                            else:
                                cell_content.append(f"{symbol}({prob_val:.6f})")
                        
                        if line_idx < len(cell_content):
                            multi_line_row += cell_content[line_idx].center(cell_width)
                        else:
                            multi_line_row += ' ' * cell_width
            
            print(multi_line_row)
        
        # Print separator between rows
        print('-' * (10 + cell_width * n))

else:
    print("No valid parse found.")
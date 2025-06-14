# This file will plot a given sparse matrix in .mtx file.
# All non zeroes indices are plotted and all M[i, j] having same value
# have same color.

import sys
import matplotlib.pyplot as plt

colors =  'rgbcmyk'

matrix_filename = '../../matrices/' + sys.argv[1] + '.mtx'
output_filename = '../../matrices/' + sys.argv[1] + '.png'

lines = [line.rstrip('\n').strip() for line in open(matrix_filename)]
first_line_not_percent = 0

while True:
    if lines[first_line_not_percent][0] == '%':
        first_line_not_percent = first_line_not_percent + 1
    else:
        break

x = []
y = []
c = []

mapping = {}

for line in lines[first_line_not_percent + 1:]:
    [cx, cy, cc] = map(lambda x : float(x), line.split())
    if cc not in mapping:
        mapping[cc] = len(mapping)
    x.append(cx)
    y.append(cy)
    c.append(colors[mapping[cc] % len(colors)])

plt.scatter(x, y, s = 4, c = c)
plt.savefig(output_filename)

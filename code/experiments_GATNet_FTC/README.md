## experiments_a.sh
- find the new GATNet 
- Candidate GAT model: GATConv, GATv2Conv
- Candidate activation functions: hardtanh, mish, hardswish, silu, tanh, hardsigmoid, rrelu, leaky_relu, celu, selu, elu, relu6, tanhshrink
- note that the log file is very large and hard to find the better result:
  1. get_line_number.py
  2. get_line_content.sh

## get_line_number.py
- Extract the floating-point number following the pattern "node    precision | micro:" and generate a grouped ranking representation.

## get_line_content.sh
- Output the FTC results from the manually selected line numbers in the log to a specified file for review.

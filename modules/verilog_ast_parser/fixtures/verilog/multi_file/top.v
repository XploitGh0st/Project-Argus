// Multi-file design, file 1/2: top-level instantiating alu.
module soc_top(input [3:0] a, input [3:0] b, output [3:0] y);
  alu u_alu(.a(a), .b(b), .y(y));
endmodule

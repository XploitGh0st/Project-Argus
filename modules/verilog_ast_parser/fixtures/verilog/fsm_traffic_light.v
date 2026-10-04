// Traffic light FSM: 0=RED, 1=GREEN, 2=YELLOW.
module traffic(input clk, input rst, input start, output reg [1:0] light);
  reg [1:0] state;

  always @(posedge clk) begin
    if (rst) state <= 2'd0;
    else begin
      case (state)
        2'd0: if (start) state <= 2'd1;
        2'd1: state <= 2'd2;
        2'd2: state <= 2'd0;
      endcase
    end
  end

  always @(*) begin
    if (state == 2'd2) light = 2'd3;
    else light = state;
  end
endmodule

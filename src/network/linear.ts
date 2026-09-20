import { matmul } from "./common.ts";

export class Linear {
    constructor(
        private readonly weights: number[][],
        private readonly bias: number[]
    ) {}

    forward(x: number[][]): number[][] {
        const output = matmul(x, this.weights);

        return output.map((row) => {
            return row.map((value, index) => {
                return value + this.bias[index];
            })
        })
    }
}

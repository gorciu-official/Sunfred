import { matmul } from "../common.ts";

export class ScaledDotProductAttention {
    constructor(private readonly keyDimension: number) {}

    private softmax(values: number[]): number[] {
        const max = Math.max(...values);
    
        const exponentials = values.map(
            x => Math.exp(x - max)
        );
    
        const sum = exponentials.reduce(
            (a, b) => a + b,
            0
        );
    
        return exponentials.map(
            x => x / sum
        );
    }

    private transpose(A: number[][]): number[][] {
        if (A.length == 0) return [];

        const rows = A.length;
        const cols = A[0].length;

        return Array.from({ length: cols }, (_, j) =>
            Array.from({ length: rows }, (_, i) => A[i][j])
        );
    }

    private scale(A: number[][], scalar: number) {
        return A.map(row => row.map(x => x * scalar));
    }

    forward(query: number[][], key: number[][], value: number[][]) {
        const scores = matmul(query, this.transpose(key));
        const scaledScores = this.scale(scores, 1 / Math.sqrt(this.keyDimension));
        const weights = scaledScores.map((row) => {
            return this.softmax(row);
        });
        return matmul(weights, value);
    }
}

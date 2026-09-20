import { randomMatrix, zeroVector } from "../common.ts";
import { Linear } from "../linear.ts";

export class FeedForward {
    private readonly linear1: Linear;
    private readonly linear2: Linear;

    constructor(
        embeddingDim: number,
        hiddenDim: number
    ) {
        this.linear1 = new Linear(
            randomMatrix(embeddingDim, hiddenDim),
            zeroVector(hiddenDim)
        );

        this.linear2 = new Linear(
            randomMatrix(hiddenDim, embeddingDim),
            zeroVector(embeddingDim)
        );
    }

    forward(x: number[][]): number[][] {
        const hidden = this.linear1.forward(x);

        const activated = hidden.map(row =>
            row.map(value => Math.max(0, value))
        );

        return this.linear2.forward(activated);
    }
}

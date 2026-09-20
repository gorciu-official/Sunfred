import { assert } from "@std/assert";
import { randomMatrix } from "../common.ts";
import { Linear } from "../linear.ts";
import { ScaledDotProductAttention } from "./scaled-dot-product.ts";

function splitHeads(
    x: number[][],
    numHeads: number
): number[][][] {
    const embeddingDim = x[0].length;
    const headDim = embeddingDim / numHeads;

    if (!Number.isInteger(headDim)) {
        throw new Error(
            "embeddingDim must be divisible by numHeads"
        );
    }

    const heads: number[][][] = [];

    for (let head = 0; head < numHeads; head++) {
        const start = head * headDim;
        const end = start + headDim;

        heads.push(
            x.map(row => row.slice(start, end))
        );
    }

    return heads;
}

function concatHeads(heads: number[][][]): number[][] {
    if (heads.length === 0) {
        return [];
    }

    const seqLen = heads[0].length;

    return Array.from(
        { length: seqLen },
        (_, rowIndex) =>
            heads.flatMap(head => head[rowIndex])
    );
}

export class MultiHeadAttention {
    private readonly query: Linear;
    private readonly key: Linear;
    private readonly value: Linear;
    private readonly output: Linear;

    private readonly attention: ScaledDotProductAttention;

    constructor(
        private readonly embeddingDim: number,
        private readonly numHeads: number
    ) {
        assert(embeddingDim % numHeads == 0);

        this.query = new Linear(
            randomMatrix(embeddingDim, embeddingDim),
            Array(embeddingDim).fill(0)
        );

        this.key = new Linear(
            randomMatrix(embeddingDim, embeddingDim),
            Array(embeddingDim).fill(0)
        );

        this.value = new Linear(
            randomMatrix(embeddingDim, embeddingDim),
            Array(embeddingDim).fill(0)
        );

        this.output = new Linear(
            randomMatrix(embeddingDim, embeddingDim),
            Array(embeddingDim).fill(0)
        );

        this.attention = new ScaledDotProductAttention(this.embeddingDim / numHeads);
    }

    forward(x: number[][]): number[][] {
        const q = this.query.forward(x);
        const k = this.key.forward(x);
        const v = this.value.forward(x);

        const qHeads = splitHeads(q, this.numHeads);
        const kHeads = splitHeads(k, this.numHeads);
        const vHeads = splitHeads(v, this.numHeads);

        const heads: number[][][] = [];

        for (let i = 0; i < this.numHeads; i++) {
            const head = this.attention.forward(
                qHeads[i],
                kHeads[i],
                vHeads[i]
            );

            heads.push(head);
        }

        const concatenated = concatHeads(heads);

        return this.output.forward(concatenated);
    }
}

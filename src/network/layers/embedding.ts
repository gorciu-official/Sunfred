import { randomMatrix } from "../common.ts";

export class EmbeddingLayer {
    private embeddings: number[][];

    constructor(private embeddingDim: number, private vocabSize: number) {
        this.embeddings = randomMatrix(this.vocabSize, this.embeddingDim); 
    }

    public forward(tokenIds: number[]): number[][] {
        const output: number[][] = [];
        for (const tokenId of tokenIds)
            output.push(this.embeddings[tokenId]);
        return output;
    }
}

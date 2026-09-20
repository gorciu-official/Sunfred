import { MultiHeadAttention } from "./attention/multi-head.ts";
import { addMatrices } from "./common.ts";
import { FeedForward } from "./layers/feed-forward.ts";
import { LayerNorm } from "./layers/layer-norm.ts";

export class TransformerBlock {
    private readonly attention: MultiHeadAttention;
    private readonly attentionNorm: LayerNorm;

    private readonly feedForward: FeedForward;
    private readonly feedForwardNorm: LayerNorm;

    constructor(
        embeddingDim: number,
        hiddenDim: number,
        numHeads: number
    ) {
        this.attention = new MultiHeadAttention(embeddingDim, numHeads);
        this.attentionNorm = new LayerNorm();

        this.feedForward = new FeedForward(embeddingDim, hiddenDim);
        this.feedForwardNorm = new LayerNorm();
    }

    forward(x: number[][]): number[][] {
        const attentionOut = this.attention.forward(x);

        const attentionResidual = addMatrices(x, attentionOut);
        const attentionNormalized = this.attentionNorm.forward(attentionResidual);
        const feedForwardOut = this.feedForward.forward(attentionNormalized);
        const feedForwardResidual = addMatrices(attentionNormalized, feedForwardOut);

        return this.feedForwardNorm.forward(feedForwardResidual);
    }
}

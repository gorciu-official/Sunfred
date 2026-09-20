import { MultiHeadAttention } from "./attention/multi-head.ts";
import { addMatrices } from "./common.ts";
import { EmbeddingLayer } from "./layers/embedding.ts";
import { FeedForward } from "./layers/feed-forward.ts";
import { LayerNorm } from "./layers/layer-norm.ts";
import { PositionalEncoding } from "./layers/positional-encoding.ts";
import { Tokenizer } from "./tokenizer.ts";

class Network {
    private tokenizer: Tokenizer;
    private embeddingLayer: EmbeddingLayer;
    private attentionLayer: MultiHeadAttention;
    private positionalEncoding: PositionalEncoding; 
    private layerNorm: LayerNorm;
    private feedForward: FeedForward;

    constructor(private vocab: string[], private embeddingDim: number, private hiddenDim: number) {
        this.tokenizer = new Tokenizer(this.vocab);
        this.embeddingLayer = new EmbeddingLayer(this.embeddingDim, this.vocab.length);
        this.attentionLayer = new MultiHeadAttention(this.embeddingDim, this.embeddingDim);
        this.positionalEncoding = new PositionalEncoding(this.embeddingDim);
        this.layerNorm = new LayerNorm();
        this.feedForward = new FeedForward(this.embeddingDim, this.hiddenDim);
    };

    generateResponse(question: string) { 
        const tokenized = this.tokenizer.tokenize(question);
        console.log(`tokenized question: ${JSON.stringify(tokenized)}`);

        const embeddings = this.embeddingLayer.forward(tokenized);
        console.log(`embeddings: ${JSON.stringify(embeddings)}`);

        const positioned = this.positionalEncoding.forward(embeddings);
        console.log(`positional encoding output: ${JSON.stringify(positioned)}`);

        const attentionOut = this.attentionLayer.forward(positioned);
        console.log(`self attention output: ${JSON.stringify(attentionOut)}`);

        const added = addMatrices(positioned, attentionOut);
        console.log(`added: ${JSON.stringify(added)}`);

        const normalized = this.layerNorm.forward(added);
        console.log(`normalized: ${JSON.stringify(normalized)}`);

        const feedForwardOut = this.feedForward.forward(normalized);
        console.log(`feed forward: ${JSON.stringify(feedForwardOut)}`);

        const secondAdded = addMatrices(normalized, feedForwardOut);
        console.log(`second added: ${JSON.stringify(secondAdded)}`);

        const output = this.layerNorm.forward(secondAdded);
        console.log(`output: ${JSON.stringify(output)}`);

        return question;
    }
}

const n = new Network([',', '!', 'hello', 'world'], 8, 32);
console.log('model response: ' + n.generateResponse("hello, world!"));
